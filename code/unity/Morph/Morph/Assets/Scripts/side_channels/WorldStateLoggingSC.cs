using System;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;


public class WorldStateLoggingSC : SideChannel
{
    public WorldStateLoggingSC()
    {
        ChannelId = new Guid("f5a93c6c-3806-11eb-ab0a-d8cb8a18a539");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
    }

    public void SendStateInfo(string obj, float value)
    {
        string stringToSend = obj + ":" + value;

        using (var msgOut = new OutgoingMessage())
        {
            msgOut.WriteString(stringToSend);
            QueueMessageToSend(msgOut);
        }
    }
}