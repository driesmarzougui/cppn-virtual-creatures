using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;


/// <summary>
/// A simple free camera to be added to a Unity game object.
/// 
/// Keys:
///	zqsd / arrows	- movement
///	a/e 			- up/down (local space)
///	r/f 			- up/down (world space)
///	pageup/pagedown	- up/down (world space)
///	hold shift		- enable fast movement mode
///	right mouse  	- enable free look
///	mouse			- free look / rotation
///     
/// </summary>
public class CameraController : MonoBehaviour
{
    /// <summary>
    /// Normal speed of camera movement.
    /// </summary>
    public float movementSpeed = 10f;

    /// <summary>
    /// Speed of camera movement when shift is held down,
    /// </summary>
    public float fastMovementSpeed = 100f;

    /// <summary>
    /// Sensitivity for free look.
    /// </summary>
    public float freeLookSensitivity = 3f;

    /// <summary>
    /// Amount to zoom the camera when using the mouse wheel.
    /// </summary>
    public float zoomSensitivity = 10f;

    /// <summary>
    /// Amount to zoom the camera when using the mouse wheel (fast mode).
    /// </summary>
    public float fastZoomSensitivity = 50f;

    /// <summary>
    /// Set to true when free looking (on right mouse button).
    /// </summary>
    private bool looking = false;

    public bool shouldRotate = true;

    // The target we are following
    private Transform agent;

    // The distance in the x-z plane to the target
    public float distance = 10.0f;

    // the height we want the camera to be above the target
    public float height = 5.0f;

    // How much we
    public float heightDamping = 2.0f;
    public float rotationDamping = 3.0f;
    float wantedRotationAngle;
    float wantedHeight;
    float currentRotationAngle;
    float currentHeight;
    Quaternion currentRotation;

    void LateUpdate()
    {
        if (PersistantWorldManager.Instance.SelectedAgent)
        {
            agent = PersistantWorldManager.Instance.SelectedAgent.transform;

            // Calculate the current rotation angles
            wantedRotationAngle = agent.eulerAngles.y;
            wantedHeight = agent.position.y + height;
            currentRotationAngle = transform.eulerAngles.y;
            currentHeight = transform.position.y;
            // Damp the rotation around the y-axis
            currentRotationAngle =
                Mathf.LerpAngle(currentRotationAngle, wantedRotationAngle, rotationDamping * Time.deltaTime);
            // Damp the height
            currentHeight = Mathf.Lerp(currentHeight, wantedHeight, heightDamping * Time.deltaTime);
            // Convert the angle into a rotation
            currentRotation = Quaternion.Euler(0, currentRotationAngle, 0);
            // Set the position of the camera on the x-z plane to:
            // distance meters behind the target
            transform.position = agent.position;
            transform.position -= currentRotation * Vector3.forward * distance;
            // Set the height of the camera
            transform.position = new Vector3(transform.position.x, currentHeight, transform.position.z);
            // Always look at the target
            if (shouldRotate)
                transform.LookAt(agent);
        }
        else
        {
            var fastMode = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
            var movementSpeed = fastMode ? this.fastMovementSpeed : this.movementSpeed;

            if (Input.GetKey(KeyCode.Q) || Input.GetKey(KeyCode.LeftArrow))
            {
                transform.position = transform.position + (-transform.right * (movementSpeed * Time.deltaTime));
            }

            if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow))
            {
                transform.position = transform.position + (transform.right * (movementSpeed * Time.deltaTime));
            }

            if (Input.GetKey(KeyCode.Z) || Input.GetKey(KeyCode.UpArrow))
            {
                transform.position = transform.position + (transform.forward * (movementSpeed * Time.deltaTime));
            }

            if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))
            {
                transform.position = transform.position + (-transform.forward * (movementSpeed * Time.deltaTime));
            }

            if (Input.GetKey(KeyCode.A))
            {
                transform.position = transform.position + (transform.up * (movementSpeed * Time.deltaTime));
            }

            if (Input.GetKey(KeyCode.E))
            {
                transform.position = transform.position + (-transform.up * (movementSpeed * Time.deltaTime));
            }

            if (Input.GetKey(KeyCode.R) || Input.GetKey(KeyCode.PageUp))
            {
                transform.position = transform.position + (Vector3.up * (movementSpeed * Time.deltaTime));
            }

            if (Input.GetKey(KeyCode.F) || Input.GetKey(KeyCode.PageDown))
            {
                transform.position = transform.position + (-Vector3.up * (movementSpeed * Time.deltaTime));
            }

            if (looking)
            {
                float newRotationX = transform.localEulerAngles.y + Input.GetAxis("Mouse X") * freeLookSensitivity;
                float newRotationY = transform.localEulerAngles.x - Input.GetAxis("Mouse Y") * freeLookSensitivity;
                transform.localEulerAngles = new Vector3(newRotationY, newRotationX, 0f);
            }

            float axis = Input.GetAxis("Mouse ScrollWheel");
            if (axis != 0)
            {
                var zoomSensitivity = fastMode ? this.fastZoomSensitivity : this.zoomSensitivity;
                transform.position = transform.position + transform.forward * (axis * zoomSensitivity);
            }

            if (Input.GetKeyDown(KeyCode.Mouse1))
            {
                StartLooking();
            }
            else if (Input.GetKeyUp(KeyCode.Mouse1))
            {
                StopLooking();
            }
        }

        if (Input.GetKeyDown(KeyCode.Mouse0))
        {
            // Focus on certain agent
            FocusAgent();
        }
    }

    void OnDisable()
    {
        StopLooking();
    }

    /// <summary>
    /// Enable free looking.
    /// </summary>
    public void StartLooking()
    {
        looking = true;
        Cursor.visible = false;
        Cursor.lockState = CursorLockMode.Locked;
    }

    /// <summary>
    /// Disable free looking.
    /// </summary>
    public void StopLooking()
    {
        looking = false;
        Cursor.visible = true;
        Cursor.lockState = CursorLockMode.None;
    }

    public void FocusAgent()
    {
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hitInfo = new RaycastHit();
        bool hit = Physics.Raycast(ray, out hitInfo);

        if (hit && hitInfo.transform.gameObject.CompareTag("agent"))
        {
            PersistantWorldManager.Instance.SelectedAgent = hitInfo.transform.gameObject;
        }
        else
        {
            PersistantWorldManager.Instance.SelectedAgent = null;
        }
    }
}