using System.Collections;
using System.Collections.Generic;
using Agent.Morphology;
using UnityEngine;

public class SideChannelEventHandler : MonoBehaviour
{
    public GameObject agentBlock;
    public GameObject mAgent;

    public GameObject world;

    public void AddGenome(string genomeId)
    {
        /*  Deprecated
        GameObject agent = Instantiate(agentBlock, Vector3.zero, Quaternion.identity, world.transform);
        agent.GetComponent<AgentController>().AddToWorld(genomeId);
        agent.GetComponent<AgentState>().Reset();
        //agent.GetComponent<AgentBrain>().OnEpisodeBegin();
        */
    }

    public void CreateMorphology(string morphString)
    {
        MorphInfo morphInfo = JsonUtility.FromJson<MorphInfo>(morphString);

        GameObject agent = Instantiate(mAgent, Vector3.zero, Quaternion.identity, world.transform);

        AgentMorphBuilder agentMB = agent.GetComponent<AgentMorphBuilder>();
        agentMB.BuildMorphology(morphInfo);

        agent.GetComponent<AgentState>().Reset();

        agent.GetComponent<AgentController>().MAddToWorld(morphInfo.genomeId.ToString(), agentMB.Joints,
            agentMB.SensorBlocks, agentMB.BrainBlockGO, agentMB.SizePerDimension);
    }
}