using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Controller of the energy source object that spawns when Agent's die.
/// </summary>
public class EnergyCapsuleController : MonoBehaviour
{
    private float origFoodEnery;
    private float foodEnergy;
    private float drinkEnergy;

    private HashSet<AgentState> OnMe = new HashSet<AgentState>();


    public void SetEnergy(float fE, float dE)
    {
        foodEnergy = fE;
        drinkEnergy = dE;
        origFoodEnery = fE;
    }

    public Vector2 GetEaten(Vector2 energyToTake)
    {
        float foodEnergyToGive = Mathf.Min(this.foodEnergy, energyToTake[0]);
        float drinkEnergyToGive = Mathf.Min(this.drinkEnergy, energyToTake[1]);

        foodEnergy -= foodEnergyToGive;
        drinkEnergy -= drinkEnergyToGive;

        return new Vector2(foodEnergyToGive, drinkEnergyToGive);
    }


    private void Update()
    {
        foodEnergy = Mathf.Max(0f, foodEnergy - 0.001f);
        drinkEnergy = Mathf.Max(0f, drinkEnergy - 0.001f);


        if (foodEnergy <= float.Epsilon)
        {
            foreach (var agentState in OnMe)
            {
                if (agentState == null)
                {
                    continue;
                }

                agentState.RemoveFood(gameObject);
            }

            Destroy(gameObject);
        }
    }

    private void FixedUpdate()
    {
        transform.localScale = foodEnergy / origFoodEnery * 0.7f * Vector3.one;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.attachedRigidbody && other.gameObject.CompareTag("agent"))
        {
            AgentState agentState = other.gameObject.GetComponent<AgentState>();
            agentState.AddFood(this.gameObject);
            OnMe.Add(agentState);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.attachedRigidbody && other.gameObject.CompareTag("agent"))
        {
            AgentState agentState = other.gameObject.GetComponent<AgentState>();
            agentState.RemoveFood(this.gameObject);
            OnMe.Remove(agentState);
        }
    }
}