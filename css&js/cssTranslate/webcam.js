// Function to handle the stopping of the camera
  function stopCamera() {
    fetch('/stop_camera');  // Trigger the server-side route to stop the camera
  }

  // Add an event listener for when the page is about to unload
  window.onbeforeunload = stopCamera;
