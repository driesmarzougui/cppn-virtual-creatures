using System;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class MorphCreatorSC : SideChannel
{
    
    private SideChannelEventHandler sceh;

    public MorphCreatorSC(SideChannelEventHandler sceh)
    {
        ChannelId = new Guid("e910421e-4dd9-11eb-a9e8-e3175c87fdbf");
        this.sceh = sceh;
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        string receivedString = msg.ReadString();

        Debug.Log("Received morph creation request!");
        sceh.CreateMorphology(receivedString);
        
        SendAck();
    }

    public void SendAck()
    {
        string stringToSend = "agent-morph-creation-ready";
        Debug.Log("Sending morph creation ACK!");
        using (var msgOut = new OutgoingMessage())
        {
            msgOut.WriteString(stringToSend);
            QueueMessageToSend(msgOut);
        }
    }
}