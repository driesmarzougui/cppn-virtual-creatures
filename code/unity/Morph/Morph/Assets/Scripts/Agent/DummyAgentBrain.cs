using System.Collections;
using System.Collections.Generic;
using Agent.Utils;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class DummyAgentBrain : Unity.MLAgents.Agent 
{
    public override void OnEpisodeBegin()
    {
    }

    public override void CollectObservations(VectorSensor sensor)
    {
    }

    private void CalculateRewards()
    {
    }

    public override void OnActionReceived(float[] vectorAction)
    {
    }

    public override void Heuristic(float[] actionsOut)
    {
    }
    
}
