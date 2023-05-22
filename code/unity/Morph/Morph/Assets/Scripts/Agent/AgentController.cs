using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Numerics;
using Agent.Utils;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.UI;
using Quaternion = UnityEngine.Quaternion;
using Random = UnityEngine.Random;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;

/// <summary>
/// Main agent controller, handles
///     - (inter-)agent actions
///     - Agent UI
///
/// Interacts with Agent'
///     - brain:                episode handling
///     - state:                state handling
///     - energyController:     energy handling
///     - genes:                gene dependent actions (e.g. movement speed) 
/// </summary>
public class AgentController : MonoBehaviour
{
    // UI
    private Slider foodEnergySlider;

    // Components
    public AgentGenes genes;
    public AgentState state;
    public AgentBrain brain;
    public AgentEnergyController energyController;

    public VegetationSpawner vegetationSpawner;

    // Movement
    private List<ConfigurableJoint> joints = new List<ConfigurableJoint>();
    public List<ConfigurableJoint> Joints => joints;

    // Sensors
    private List<GameObject> sensorBlocks = new List<GameObject>();
    public List<GameObject> SensorBlocks => sensorBlocks;

    // Morphology
    private Vector3 agentSizePerDimension;

    public Vector3 AgentSizePerDimension => agentSizePerDimension;

    private GameObject brainBlockGO;
    public Rigidbody brainBlockRB;

    public GameObject BrainBlockGO
    {
        get => brainBlockGO;
        set => brainBlockGO = value;
    }

    void Awake()
    {
        genes = GetComponent<AgentGenes>();
        state = GetComponent<AgentState>();
        energyController = GetComponent<AgentEnergyController>();
        vegetationSpawner = GetComponent<VegetationSpawner>();
    }

    IEnumerator InitialiseBrain()
    {
        // Temporarily do this with a coroutine to make sure the agent lies still before starting
        yield return new WaitForSeconds(2f);

        state.IsValid = true;
        brain = gameObject.AddComponent<AgentBrain>();
        brain.MaxStep = 0;
        DecisionRequester decisionRequester = gameObject.AddComponent<DecisionRequester>();
        decisionRequester.DecisionPeriod = 5;
        decisionRequester.TakeActionsBetweenDecisions = true;
    }

    public void MAddToWorld(string genomeID, List<ConfigurableJoint> cJoints, List<GameObject> sBlocks,
        GameObject brainBlock, Vector3 asPerDim)
    {
        Debug.Log($"MAddToWorld for {genomeID} called");
        joints = cJoints;
        sensorBlocks = sBlocks;
        brainBlockGO = brainBlock;
        agentSizePerDimension = asPerDim;
        brainBlockRB = brainBlockGO.GetComponent<Rigidbody>();
        genes.GenomeId = genomeID;
        gameObject.GetComponent<BehaviorParameters>().BehaviorName = genes.GenomeId;

        foodEnergySlider = brainBlockGO.GetComponent<BrainBlockController>().foodEnergySlider;
        foodEnergySlider.maxValue = genes.MaxFoodEnergy;
        StartCoroutine(InitialiseBrain());
    }

    // Body Control ----------------------------------------------------------------------------------------------------
    public void SetJointAngles(ConfigurableJoint joint, float x, float y, float z)
    {
        float lax = joint.lowAngularXLimit.limit;
        float hax = joint.highAngularXLimit.limit;
        float ayl = joint.angularYLimit.limit;
        float azl = joint.angularZLimit.limit;

        x = Mathf.Clamp(x, -0.95f, 0.95f);
        y = Mathf.Clamp(y, -0.95f, 0.95f);
        z = Mathf.Clamp(z, -0.95f, 0.95f);

        Vector3 eulerAngles = Vector3.zero;
        eulerAngles.x = x <= 0 ? lax * Mathf.Abs(x) : hax * x;
        eulerAngles.y = y * ayl;
        eulerAngles.z = z * azl;

        RotationUtils.SetTargetRotationLocal(joint, Quaternion.Euler(eulerAngles));
    }

    private void CheckValid()
    {
        if (state.IsValid && brain.numLives == 1)
        {
            float brainSpeed = brainBlockRB.velocity.magnitude;
            if (brainSpeed > 50f)
            {
                Debug.Log($"Morphology probably exploded! Brain speed: {brainSpeed}");
                state.IsValid = false;
                brain.EndEpisode();
            }
        }
    }

    public void Eat()
    {
        if (state.Food.Count > 0)
        {
            float energyGained = state.Eat(state.Food.ElementAt(0));
            if (energyGained > 0f)
            {
                brain.AddReward(energyGained);
            }
        }
        else
        {
            energyController.TryEatingEnergyConsumption();
        }
    }

    // Death -----------------------------------------------------------------------------------------------------------
    public void Die()
    {
        Debug.Log($"Die() for {genes.genomeId} called!");
        DestroyImmediate(GetComponent<DecisionRequester>());
        DestroyImmediate(brain);
        state.Reset();
        StopAllCoroutines();
        //PersistantWorldManager.Instance.RemoveAgent(state.NewAgent);
        DestroyImmediate(gameObject);
    }

    // Utils -----------------------------------------------------------------------------------------------------------

    private float CalcDistanceToClosestPlant(Vector3 position)
    {
        int numPlantsLeft = vegetationSpawner.Vegetation.transform.childCount;
        float minDistance = 1000f;
        float distance;
        for (int i = 0; i < numPlantsLeft; i++)
        {
            distance = Vector3.Distance(position, vegetationSpawner.Vegetation.transform.GetChild(i).position);
            if (distance < minDistance)
            {
                minDistance = distance;
            }
        }

        return minDistance;
    }

    private void AddFinalReward()
    {
        // Calculate distance to closest vegetation and return inverse as reward
        int numPlantsLeft = vegetationSpawner.Vegetation.transform.childCount;
        if (numPlantsLeft > 0)
        {
            Vector3 position = brainBlockGO.transform.position;
            float originalDistance = CalcDistanceToClosestPlant(Vector3.zero);
            float currentDistance = CalcDistanceToClosestPlant(position);
            float reward = Mathf.Clamp(1 - currentDistance / originalDistance, 0f, 1f);
            brain.AddReward(reward);
        }
        else
        {
            brain.AddReward(10f);
        }
    }

    private void UpdateEnergy()
    {
        if (!energyController.ConstantUpdate())
        {
            // Death
            AddFinalReward();
            brain.EndEpisode();
        }
    }


    public void UpdateUI()
    {
        foodEnergySlider.value = state.FoodEnergy;
    }

    // -----------------------------------------------------------------------------------------------------------------
    private void Update()
    {
        //LocaliseAgentInWorld();
        //UpdateParent();
        UpdateUI();
    }

    private void FixedUpdate()
    {
        if (brain)
        {
            UpdateEnergy();
            CheckValid();
        }
    }
}