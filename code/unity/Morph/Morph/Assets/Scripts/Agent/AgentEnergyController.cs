using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.Windows.WebCam;

/// <summary>
/// Handles agent' energy acquisition and consumption.
/// </summary>
public class AgentEnergyController : MonoBehaviour
{
    public float EPS = 0.00001f;

    private AgentState state;

    private void Awake()
    {
        state = GetComponent<AgentState>();
    }


    public void TryEatingEnergyConsumption()
    {
        state.FoodEnergy -= 0.0001f;
    }

    public bool ConstantUpdate()
    {
        state.FoodEnergy -= 0.0002f;
        return state.FoodEnergy > EPS;
    }


    public void SpawnEnergyCapsule()
    {
    }
}