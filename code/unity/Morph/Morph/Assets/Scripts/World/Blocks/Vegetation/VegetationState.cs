using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Represents the vegetation's state, which is currently just its energy.
/// </summary>
public class VegetationState : MonoBehaviour
{
    private float energy = 1f;

    public float GetEnergy()
    {
        return energy;
    }

    public void DeltaEnergy(float delta)
    {
        energy += delta;
    }

    public bool WillSpread(float minSpreadEnergy, float probability)
    {
        // todo: add cooldown timer
        return energy > minSpreadEnergy && Random.Range(0.0f, 1.0f) < probability;
    }

    public float CanGrow(float maxEnergy)
    {
        return energy < maxEnergy ? maxEnergy - energy : 0.0f; 
    }
}
