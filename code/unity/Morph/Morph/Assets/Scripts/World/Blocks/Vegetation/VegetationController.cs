using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using UnityEditor;
using UnityEngine;
using Random = UnityEngine.Random;

/// <summary>
/// Main vegetation controller, handles:
///     - Getting eaten
///     - Growing
///     - Spreading
/// </summary>
public class VegetationController : MonoBehaviour
{
    public GameObject vegetationBlock;
    public Vector3 maxSize;

    private int worldX;
    private int worldZ;
    private Vector3 position;

    private VegetationGenes genes;
    private VegetationState state;

    private HashSet<AgentState> OnMe = new HashSet<AgentState>();

    private void Start()
    {
        position = this.transform.position;
        worldX = (int) position.x + PersistantWorldManager.Instance.width / 2;
        worldZ = (int) position.z + PersistantWorldManager.Instance.depth / 2;

        genes = this.GetComponent<VegetationGenes>();
        state = this.GetComponent<VegetationState>();
    }

    public float GetEaten(float energyTaken)
    {
        energyTaken = Mathf.Min(energyTaken, state.GetEnergy());
        state.DeltaEnergy(-energyTaken);
        return energyTaken;
    }

    private void Spread()
    {
        if (state.WillSpread(genes.minSpreadEnergy, genes.probability) &&
            PersistantWorldManager.Instance.VegetationCanSpread())
        {
            // Pick random position offset within spread radius
            int xOffset = Random.Range(-genes.spreadRadius, genes.spreadRadius);
            int zOffset = Random.Range(-genes.spreadRadius, genes.spreadRadius);

            // Check if target position is within world 
            if (0 <= worldZ + zOffset && worldZ + zOffset < PersistantWorldManager.Instance.depth
                                      && 0 <= worldX + xOffset &&
                                      worldX + xOffset < PersistantWorldManager.Instance.width)
            {
                // Check if new location has earth underneath
                GameObject targetBlock = PersistantWorldManager.Instance.world[worldZ + zOffset, worldX + xOffset];
                if (targetBlock.CompareTag("earthBlock"))
                {
                    // Check if new location is free
                    if (targetBlock.transform.childCount == 0)
                    {
                        if (PersistantWorldManager.Instance.AddVegetation())
                        {
                            GameObject go = Instantiate(vegetationBlock,
                                new Vector3(position.x + xOffset, position.y, position.z + zOffset),
                                Quaternion.identity, targetBlock.transform);

                            // Pass genes to child
                            go.GetComponent<VegetationGenes>().Inherit(genes);
                        }
                    }
                }
            }
        }
    }

    private void Grow()
    {
        float energyGap = Mathf.Min(state.CanGrow(genes.maxEnergy), genes.energyGainPerTimestep);
        state.DeltaEnergy(energyGap);
    }

    // Update is called once per frame
    void Update()
    {
        if (state.GetEnergy() <= Single.Epsilon)
        {
            PersistantWorldManager.Instance.RemoveVegetation();
            Destroy(gameObject);
        }

        //Spread();

        //Grow();
    }

    private void FixedUpdate()
    {
        // float energy = state.GetEnergy();
        // transform.localScale = energy * maxSize;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.attachedRigidbody && other.gameObject.CompareTag("agent"))
        {
            AgentState agentState = other.gameObject.GetComponentInParent<AgentState>();
            agentState.AddFood(this.gameObject);
            OnMe.Add(agentState);
        }
    }

    private void OnDestroy()
    {
        foreach (var agentState in OnMe)
        {
            if (agentState)
            {
                agentState.RemoveFood(this.gameObject);
            }
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.attachedRigidbody && other.gameObject.CompareTag("agent"))
        {
            AgentState agentState = other.gameObject.GetComponentInParent<AgentState>();
            agentState.RemoveFood(this.gameObject);
            OnMe.Remove(agentState);
        }
    }
}