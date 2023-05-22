using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using Agent.Utils;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using Random = UnityEngine.Random;

/// <summary>
/// Represents the agent's brain as described in https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Design-Agents.md
///
/// </summary>
public class AgentBrain : Unity.MLAgents.Agent
{
    private AgentController agentController;
    private AgentState agentState;
    private AgentGenes agentGenes;

    public int numLives;

    private Vector3 agentSizePerDimensionScale;
    private Vector3 eulerAngleScale;

    public override void Initialize()
    {
        agentController = GetComponent<AgentController>();
        agentState = GetComponent<AgentState>();
        agentGenes = GetComponent<AgentGenes>();
        agentSizePerDimensionScale = new Vector3(
            1 / agentController.AgentSizePerDimension.x,
            1 / agentController.AgentSizePerDimension.y,
            1 / agentController.AgentSizePerDimension.z);
        eulerAngleScale = Vector3.one / 180f;
    }

    public override void OnEpisodeBegin()
    {
        Debug.Log(
            $"OnEpisodeBegin called for {agentGenes.genomeId}! IsValid: {agentState.IsValid} - numLives: {numLives}");
        numLives += 1;
        if (numLives > 1)
        {
            agentController.Die();
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (agentState.IsValid && numLives == 1)
        {
            // Current local position and rotation per joint
            Vector3 jointLocalPosition;
            Vector3 jointLocalRotation;
            foreach (ConfigurableJoint cJoint in agentController.Joints)
            {
                // Add mount block's local position (in function of brain block position) 
                jointLocalPosition = cJoint.transform.position - agentController.BrainBlockGO.transform.position;
                jointLocalPosition.Scale(agentSizePerDimensionScale);

                // Add normalized joint angles
                jointLocalRotation = RotationUtils.GetJointLocalRotationEulerNormalised(cJoint);

                sensor.AddObservation(jointLocalPosition);
                sensor.AddObservation(jointLocalRotation);
            }

            // Food energy
            sensor.AddObservation(agentState.FoodEnergy);
            
            // Oscillatory signals
            sensor.AddObservation(Mathf.Sin(Time.fixedTime));
            sensor.AddObservation(Mathf.Cos(Time.fixedTime));

            // Final obligatory observation: constant bias
            sensor.AddObservation(1f);

            // EXTRA INFORMATION
            //  Current 2D Location on XZ plane
            Vector3 position = agentController.BrainBlockGO.transform.position;
            sensor.AddObservation(new Vector2(position.x, position.z));
        }
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        if (agentState.IsValid && numLives == 1)
        {
            int i = 0;
            foreach (ConfigurableJoint cJoint in agentController.Joints)
            {
                agentController.SetJointAngles(cJoint, vectorAction[i], vectorAction[i + 1], vectorAction[i + 2]);
                i += 3;
            }

            if (vectorAction[i] > 0f)
            {
                agentController.Eat();
            }
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
    }
}