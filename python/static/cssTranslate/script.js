async function getPrediction() {
    try {
        const response = await fetch('/predict', { method: 'POST' });
        const data = await response.json();
        document.getElementById('predicted-output').textContent = data.prediction || 'None';
    } catch (error) {
        console.error('Error fetching prediction:', error);
    }
}

setInterval(getPrediction, 2000);