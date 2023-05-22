using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class Registrator : MonoBehaviour
{
    private SideChannelEventHandler sceh;
    
    private AgentCreatorSC agentCreatorSC;
    private MorphCreatorSC morphCreatorSC;
    private WorldStateLoggingSC worldStateLoggingSc;
    
    public void Awake()
    {
        sceh = GetComponent<SideChannelEventHandler>();
        agentCreatorSC = new AgentCreatorSC(sceh);
        morphCreatorSC = new MorphCreatorSC(sceh);
        worldStateLoggingSc = new WorldStateLoggingSC();
        
        SideChannelManager.RegisterSideChannel(agentCreatorSC);
        SideChannelManager.RegisterSideChannel(morphCreatorSC);
        SideChannelManager.RegisterSideChannel(worldStateLoggingSc);
    }
    public void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(agentCreatorSC);
            SideChannelManager.UnregisterSideChannel(morphCreatorSC);
            SideChannelManager.UnregisterSideChannel(worldStateLoggingSc);
        }
    }

    public WorldStateLoggingSC WorldStateLoggingSc => worldStateLoggingSc;
}
