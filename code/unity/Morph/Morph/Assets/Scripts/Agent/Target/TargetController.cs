using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.PlayerLoop;
using Random = UnityEngine.Random;

public class TargetController : MonoBehaviour
{
    // Two doors experiment
    public GameObject targetA;
    public GameObject targetB;

    private GameObject target;

    public float maxNumSteps = 3000f;
    // -
    
    private AgentController agentController;

    
    private Vector3 targetLocation;

    private float originalDistanceToGoal;

    // Start is called before the first frame update
    void Start()
    {
        if (PersistantWorldManager.Instance.twoDoorsState)
        {
            target = Instantiate(targetA, targetA.transform.position, Quaternion.identity);
            targetLocation = target.transform.position;
            PersistantWorldManager.Instance.twoDoorsState = false;
        }
        else
        {
             target = Instantiate(targetB, targetB.transform.position, Quaternion.identity);
             targetLocation = target.transform.position;
             PersistantWorldManager.Instance.twoDoorsState = true;
        }
        agentController = GetComponent<AgentController>();
        originalDistanceToGoal = Vector3.Distance(agentController.BrainBlockGO.transform.position, targetLocation);
    }

    public void GoalReached()
    {
        // Target reached
        // +1 for reaching goal, minus normalized number of steps
        float reward = 1 + (1 - agentController.brain.StepCount / maxNumSteps);
        agentController.brain.SetReward(reward);
        
        agentController.brain.EndEpisode();
    }


    private void CheckProgress()
    {
        if (agentController.state.IsValid && agentController.brain.numLives == 1)
        {
            if (agentController.brain.StepCount > maxNumSteps)
            {
                float distanceToGoal =
                    Vector3.Distance(agentController.BrainBlockGO.transform.position, targetLocation);
                float reward = Mathf.Clamp(1 - distanceToGoal / originalDistanceToGoal, 0f, 0.95f);
                agentController.brain.SetReward(reward);
                
                agentController.brain.EndEpisode();
            }
        }
    }

    private void OnDestroy()
    {
        if (target)
        {
            Destroy(target);
        } 
    }

    private void FixedUpdate()
    {
        CheckProgress();
    }
}