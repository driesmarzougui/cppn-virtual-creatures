using System.Collections.Generic;
using UnityEngine;

namespace Agent.Utils
{
    public class MorphTreeNode
    {
        public Vector3Int position;
        public MorphTreeNode parent;
        public List<MorphTreeNode> children;
        public GameObject blockType;
        public GameObject block;
        public Vector3 directionFromParent;
        
        public MorphTreeNode(Vector3Int position, MorphTreeNode parent, GameObject blockType, Vector3 dirFromParent)
        {
            this.position = position;
            this.parent = parent;
            this.blockType = blockType;
            this.children = new List<MorphTreeNode>();
            this.directionFromParent = dirFromParent;
        }

        public void AddChild(MorphTreeNode child)
        {
            children.Add(child);
        }

    }
}