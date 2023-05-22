using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Analytics;

/// <summary>
/// Represents the agent's state.
/// </summary>
public class AgentState : MonoBehaviour
{
    // Control
    private HashSet<GameObject> food = new HashSet<GameObject>();

    private AgentController ac;

    private bool isValid;
    
    // Agent State
    private float healthEnergy;
    private float foodEnergy;
    private float drinkEnergy;

    private AgentGenes genes;

    private void Awake()
    {
        genes = GetComponent<AgentGenes>();
        ac = GetComponent<AgentController>();

        foodEnergy = genes.MaxFoodEnergy;
    }

    public float HealthEnergy
    {
        get => healthEnergy;
        set => healthEnergy = value;
    }


    public float FoodEnergy
    {
        get => foodEnergy;
        set => foodEnergy = value;
    }

    public float Eat(GameObject block)
    {
        if (block != null && block.CompareTag("vegetationBlock"))
        {
            VegetationController vc = block.GetComponent<VegetationController>();
            float energyGap = genes.MaxFoodEnergy - foodEnergy;

            float energyToTake = Mathf.Min(0.1f, energyGap);

            float energyTaken = vc.GetEaten(energyToTake);
            foodEnergy += energyTaken;
            return energyTaken;
        }
        return 0f;
    }

    public void AddFood(GameObject newFood)
    {
        food.Add(newFood);
    }

    public void RemoveFood(GameObject oldFood)
    {
        food.Remove(oldFood);
    }

    public HashSet<GameObject> Food => food;

    public float AgentSurvival()
    {
        return 0f;
    }

    public bool IsValid
    {
        get => isValid;
        set => isValid = value;
    }


    public void Reset()
    {
        food = new HashSet<GameObject>();
        isValid = false;
        foodEnergy = genes.MaxFoodEnergy;
    }
}