using System;
using System.Collections;
using System.Collections.Generic;
using Agent.Morphology;
using Agent.Utils;
using UnityEngine;

public class AgentMorphBuilder : MonoBehaviour
{
    public TextAsset exampleJson;
    public GameObject fillerBlock;
    public GameObject brainBlock;
    public GameObject sensorBlock;

    private float BLOCKSIZE = 1f;

    private Vector3 sizePerDimension;
    private List<ConfigurableJoint> joints = new List<ConfigurableJoint>();
    private List<GameObject> sensorBlocks = new List<GameObject>();
    private GameObject brainBlockGO;

    public Material cjBlockMaterial;

    public Vector3 SizePerDimension => sizePerDimension;
    public List<ConfigurableJoint> Joints => joints;

    public List<GameObject> SensorBlocks => sensorBlocks;

    public GameObject BrainBlockGO => brainBlockGO;


    private Quaternion GetSensorBlockRotation(BlockInfo blockInfo)
    {
        switch (blockInfo.sbd)
        {
            case 0:
                // right (X direction)
                return Quaternion.Euler(0f, 90f, 0f);
            case 1:
                // left (-X direction)
                return Quaternion.Euler(0f, 270f, 0f);
            case 2:
                // up (Y direction)
                return Quaternion.Euler(270f, 0f, 0f);
            case 3:
                // down (-Y direction)
                return Quaternion.Euler(90f, 0f, 0f);
            case 4:
                // forward (Z direction)
                return Quaternion.Euler(0f, 0f, 0f);
            case 5:
                // back (-Z direction)
                return Quaternion.Euler(0, 180f, 0f);
            default:
                return Quaternion.Euler(0f, 0f, 0f);
        }
    }

    private GameObject[,,] CreateGOMatrix(BlockInfo[] blocks, Vector3 agentSpaceDims)
    {
        Vector3 minPositionBounds = -1 * BLOCKSIZE * agentSpaceDims / 2 + (BLOCKSIZE / 2) * Vector3.one;

        // Get lowest Y position of block to calculate the y upward shift required to place the agent above ground
        float minBoundY = minPositionBounds.y;
        float lowestY = 100f;
        float currentY;
        foreach (BlockInfo block in blocks)
        {
            currentY = block.y;
            lowestY = currentY < lowestY ? currentY : lowestY;
        }

        float yShift = Mathf.Abs(minBoundY + BLOCKSIZE * lowestY) + 0.5001f; // + 0.5 as ground is at 0.5 

        // Create go matrix
        GameObject[,,] bluePrint = new GameObject[
            Mathf.CeilToInt(agentSpaceDims.x),
            Mathf.CeilToInt(agentSpaceDims.y),
            Mathf.CeilToInt(agentSpaceDims.z)
        ];

        GameObject blockType;
        Vector3 position;
        Vector3Int matrixPos;
        foreach (BlockInfo block in blocks)
        {
            switch (block.type)
            {
                case 1:
                    blockType = fillerBlock;
                    break;
                case 2:
                    blockType = sensorBlock;
                    break;
                case 3:
                    blockType = brainBlock;
                    break;
                default:
                    continue;
            }

            matrixPos = new Vector3Int(block.x, block.y, block.z);
            position = minPositionBounds + BLOCKSIZE * (Vector3) matrixPos;

            position.y += yShift;

            position *= 1.1f; // some additional spacing between blocks for physics safety
            GameObject go = Instantiate(blockType, position, Quaternion.identity, this.transform);
            bluePrint[matrixPos.x, matrixPos.y, matrixPos.z] = go;

            if (block.type == 2)
            {
                Quaternion sensorRotation = GetSensorBlockRotation(block);
                go.transform.GetChild(0).rotation = sensorRotation;
                this.sensorBlocks.Add(go);
            }
            else if (block.type == 3)
            {
                this.brainBlockGO = go;
            }
        }

        return bluePrint;
    }

    void ConnectBlocks(BlockInfo[] blocks, GameObject[,,] goMatrix)
    {
        Vector3Int matrixPos, targetMatrixPos, targetDir;
        GameObject srcGO, targetGO;
        Joint joint;
        foreach (BlockInfo block in blocks)
        {
            matrixPos = new Vector3Int(block.x, block.y, block.z);

            srcGO = goMatrix[matrixPos.x, matrixPos.y, matrixPos.z];

            foreach (JointInfo jointInfo in block.joints)
            {
                if (jointInfo.type != -1)
                {
                    targetDir = new Vector3Int(jointInfo.dx, jointInfo.dy, jointInfo.dz);
                    targetMatrixPos = matrixPos + targetDir;
                    targetGO = goMatrix[targetMatrixPos.x, targetMatrixPos.y, targetMatrixPos.z];

                    switch (jointInfo.type)
                    {
                        case 0:
                            joint = srcGO.AddComponent<FixedJoint>();
                            joint.enableCollision = false;
                            break;
                        case 1:
                            srcGO.transform.SetParent(targetGO.transform);
                            ConfigurableJoint cj = srcGO.AddComponent<ConfigurableJoint>();
                            this.joints.Add(cj);

                            cj.enablePreprocessing = false; // better for stability according to unity docs

                            // Set other material
                            srcGO.GetComponent<Renderer>().material = cjBlockMaterial;
                            Rigidbody srcGORB = srcGO.GetComponent<Rigidbody>();
                            //srcGORB.mass *= 10f; // increase mass of object by 10 to imitate "muscle" and replace box collider with sphere collider
                            Destroy(srcGO.GetComponent<BoxCollider>());
                            srcGO.AddComponent<SphereCollider>();


                            cj.enableCollision = false;

                            if (jointInfo.dy == -1)
                            {
                                // secondary axis has to be reversed as the block is connected upside down: (0, 1, 0) -> (0, -1, 0)
                                cj.secondaryAxis = -1 * cj.secondaryAxis;
                            }

                            cj.xMotion = ConfigurableJointMotion.Locked;
                            cj.yMotion = ConfigurableJointMotion.Locked;
                            cj.zMotion = ConfigurableJointMotion.Locked;
                            cj.angularXMotion = ConfigurableJointMotion.Limited;
                            cj.angularYMotion = ConfigurableJointMotion.Limited;
                            cj.angularZMotion = ConfigurableJointMotion.Limited;

                            cj.rotationDriveMode = RotationDriveMode.Slerp;
                            JointDrive slerpDrive = cj.slerpDrive;
                            slerpDrive.positionSpring = 4000f;
                            slerpDrive.positionDamper = 2000f;
                            cj.slerpDrive = slerpDrive;

                            SoftJointLimit lax = cj.lowAngularXLimit;
                            lax.limit = jointInfo.lax;
                            cj.lowAngularXLimit = lax;

                            SoftJointLimit hax = cj.highAngularXLimit;
                            hax.limit = jointInfo.hax;
                            cj.highAngularXLimit = hax;

                            SoftJointLimit ayl = cj.angularYLimit;
                            ayl.limit = jointInfo.ayl;
                            cj.angularYLimit = ayl;

                            SoftJointLimit azl = cj.angularZLimit;
                            azl.limit = jointInfo.azl;
                            cj.angularZLimit = azl;

                            if (targetGO.GetComponent<ConfigurableJoint>() == null)
                            {
                                targetGO.GetComponent<Rigidbody>().freezeRotation = true;
                            }

                            srcGORB.freezeRotation = false;

                            joint = cj;
                            break;
                        default:
                            throw new NotImplementedException();
                    }

                    joint.autoConfigureConnectedAnchor = true;
                    joint.anchor = Vector3.zero + ((Vector3) targetDir) * 0.5f; // in block space
                    joint.connectedBody = targetGO.GetComponent<Rigidbody>();
                }
            }

            srcGO.GetComponent<Rigidbody>().isKinematic = false;
        }
    }

    public void BuildMorphology(MorphInfo morphInfo)
    {
        Vector3 agentSpaceDims = new Vector3(morphInfo.agentSpaceWidth / morphInfo.agentSubSpaceWidth,
            morphInfo.agentSpaceHeight / morphInfo.agentSubSpaceHeight,
            morphInfo.agentSpaceDepth / morphInfo.agentSubSpaceDepth);

        BlockInfo[] blocks = morphInfo.blocks;

        GameObject[,,] goMatrix = CreateGOMatrix(blocks, agentSpaceDims);

        sizePerDimension = new Vector3(
            goMatrix.GetLength(0) * BLOCKSIZE, 
            goMatrix.GetLength(1) * BLOCKSIZE,
            goMatrix.GetLength(2) * BLOCKSIZE
            );
        ConnectBlocks(blocks, goMatrix);
    }

    private void Start()
    {
        /* 
        MorphInfo morphInfo = JsonUtility.FromJson<MorphInfo>(exampleJson.text);
        BuildMorphology(morphInfo);
        GetComponent<AgentController>().state.IsValid = true;
        GetComponent<AgentController>().BrainBlockGO = brainBlockGO;
        GetComponent<AgentController>().brainBlockRB = brainBlockGO.GetComponent<Rigidbody>();
        */
    }
}