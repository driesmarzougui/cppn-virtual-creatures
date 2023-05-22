using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

/// <summary>
/// Singleton class used to:
///     - have a retrievable representation of the world
///     - keep simulation state info (e.g. total agent count) and correspondingly update tensorboard logs
/// </summary>
public class PersistantWorldManager : MonoBehaviour
{
    public static PersistantWorldManager Instance { get; private set; }

    public int depth;
    public int width;
    public GameObject[,] world;
    public WorldCreator worldCreator;
    public SideChannelEventHandler sceh;

    private float agentCount;
    public int minAgentCount;
    public int maxAgentCount;

    private bool dummyAdded;

    private float earthBlockCount;
    public float vegetationCount;
    public float maxVegetationEarthRatio;

    private float naturallyBornCount;
    
    public bool twoDoorsState = true;


    private GameObject selectedAgent;

    public GameObject SelectedAgent
    {
        get => selectedAgent;
        set => selectedAgent = value;
    }


    public void AddAgent(bool naturallyBorn = false, bool dummy = false)
    {
        agentCount += 1;
        if (naturallyBorn)
        {
            naturallyBornCount += 1;
        }

        dummyAdded = dummy;
        
        UpdateTensorBoard();
    }

    public void RemoveAgent(bool naturallyBorn = false)
    {
        /*
        agentCount -= 1;
        if (naturallyBorn)
        {
            naturallyBornCount -= 1;
        }

        if (agentCount == 0 && !dummyAdded)
        {
            worldCreator.CreateAgent(dummy:true);
        }
        
        UpdateTensorBoard();
        */
    }

    public void SetEarthBlockCount(float count)
    {
        earthBlockCount = count;
    }

    public bool AddVegetation()
    {
        if (VegetationCanSpread())
        {
            vegetationCount += 1;
            UpdateTensorBoard();
            return true;
        }

        return false;
    }

    public void RemoveVegetation()
    {
        vegetationCount -= 1;
        UpdateTensorBoard();
    }

    public bool VegetationCanSpread()
    {
        return (vegetationCount + 1) / earthBlockCount < maxVegetationEarthRatio;
    }

    public float VegetationEarthRatio()
    {
        return vegetationCount / earthBlockCount;
    }

    public float AgentCount => agentCount;

    private void UpdateTensorBoard()
    {
        WorldStateLoggingSC wslsc = sceh.GetComponent<Registrator>().WorldStateLoggingSc;
        wslsc.SendStateInfo("AgentCount", agentCount);
        wslsc.SendStateInfo("NaturalArtificialBirthRatio", naturallyBornCount / agentCount);
        wslsc.SendStateInfo("VegetationRatio", VegetationEarthRatio());
    }

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }
}