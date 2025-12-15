window.onbeforeunload = function () {
    // Send a request to stop the camera
    navigator.sendBeacon('/stop_camera');
};

// Function to fetch and update the predicted word dynamically
function fetchPredictedWord() {
    fetch('/get_predicted_word')  // Call the Flask API to get the predicted word
        .then(response => response.json())
        .then(data => {
            // Update the text on the webpage with the predicted word
            document.getElementById("predicted-output").textContent = data.predicted_word;
        })
        .catch(error => {
            console.error('Error fetching predicted word:', error);
        });
}

// Call fetchPredictedWord every 1 second to update the word dynamically
setInterval(fetchPredictedWord, 1000);

// Function to fetch and update the inference time dynamically
function fetchInferenceTime() {
    fetch('/get_inference_time')  // Call the Flask API to get the inference time
        .then(response => response.json())
        .then(data => {
            // Update the text on the webpage with the inference time
            document.getElementById("predicted-time").textContent = data.inference_time;
        })
        .catch(error => {
            console.error('Error fetching inference time:', error);
        });
}

// Call fetchInferenceTime every 1 second to update the inference time dynamically
setInterval(fetchInferenceTime, 1000);

// Listen for the backspace key press
document.addEventListener('keydown', function(event) {
    if (event.key === 'Backspace') {
        fetch('/delete_last_character', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log('Last character deleted');
            })
            .catch(error => {
                console.error('Error deleting character:', error);
            });
    }
});
