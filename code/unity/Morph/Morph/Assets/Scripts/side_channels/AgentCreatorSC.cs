using System;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;


public class AgentCreatorSC : SideChannel
{
    private SideChannelEventHandler sceh;
    private bool agentsInitialized;

    public AgentCreatorSC(SideChannelEventHandler sceh)
    {
        ChannelId = new Guid("cb5d3762-3322-11eb-9dc2-417b92aeb2c2");
        this.sceh = sceh;
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        string receivedString = msg.ReadString();
        string[] genomeIDs = receivedString.Split(';');
        foreach (string genomeID in genomeIDs)
        {
            sceh.AddGenome(genomeID);
        }

        SendAck();
        agentsInitialized = true;
    }

    public void SendAck()
    {
        string stringToSend;
        if (!agentsInitialized)
        {
            stringToSend = "init-agents-ready";
        }
        else
        {
            stringToSend = "new-agents-ready";
        }

        using (var msgOut = new OutgoingMessage())
        {
            msgOut.WriteString(stringToSend);
            QueueMessageToSend(msgOut);
        }
    }
}