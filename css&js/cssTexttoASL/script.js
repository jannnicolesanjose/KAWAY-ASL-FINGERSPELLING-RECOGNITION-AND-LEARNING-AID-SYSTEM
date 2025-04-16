function displayASLWordImage(word) {
    const letterImageContainer = document.getElementById("letter-image");
    letterImageContainer.innerHTML = '';

    const errorMessageContainer = document.createElement('div');
    errorMessageContainer.id = 'error-message';

    // Using regular expression to only accept an input of word/sentences (letters only)
    if (!word.match(/^[a-zA-Z\s]+$/)) {
        errorMessageContainer.innerHTML = 'Please enter a word containing valid letters (A-Z only).';
        letterImageContainer.appendChild(errorMessageContainer);
        return;
    }

    // Convert word to uppercase for consistency
    word = word.toUpperCase();

    // Recursive divide-and-conquer processing
    function processLetters(chars) {
        // Base case: Single character
        if (chars.length === 1) {
            if (chars[0] === ' ') {
                // Create a spacer canvas for a gap if the input is a sentence (has a white space)
                const spacerCanvas = document.createElement('canvas');
                spacerCanvas.width = 70; // Width of the gap between words of the sentence
                spacerCanvas.height = 100;
                const spacerCtx = spacerCanvas.getContext('2d');
                spacerCtx.fillStyle = 'transparent';
                spacerCtx.fillRect(0, 0, spacerCanvas.width, spacerCanvas.height);
                return Promise.resolve(spacerCanvas);
            }
            return loadImage(chars[0]);
        }

        // Divide: Split into two halves
        const mid = Math.floor(chars.length / 2);
        const left = chars.slice(0, mid);
        const right = chars.slice(mid);

        // Conquer: Process each half recursively
        return Promise.all([processLetters(left), processLetters(right)]).then(([leftCanvas, rightCanvas]) => {
            // Combine: Merge the two canvases
            return mergeCanvases(leftCanvas, rightCanvas);
        });
    }

    // Start the process with the word split into characters
    processLetters([...word])
        .then((finalCanvas) => {
            // Display the combined result
            letterImageContainer.appendChild(finalCanvas);
        })
        .catch((err) => {
            errorMessageContainer.innerHTML = `Error: ${err.message}`;
            letterImageContainer.appendChild(errorMessageContainer);
        });
}

function loadImage(letter) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.src = `static/ASL/${letter}.png`;

        img.onload = function() {
            // Create a canvas for the image
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            // Set canvas size
            const letterWidth = 150;
            const letterHeight = 170;
            const textHeight = 40; // Additional space for text
            canvas.width = letterWidth;
            canvas.height = letterHeight + textHeight;

            // Draw the image on the canvas
            ctx.drawImage(img, 0, 0, letterWidth, letterHeight);

            // Add the letter as text below the image
            ctx.font = 'bold 24px Arial';
            ctx.textAlign = 'center';
            ctx.fillStyle = '#003F88';
            const textYPosition = letterHeight + textHeight - 10;
            ctx.fillText(letter, letterWidth / 2, textYPosition);

            resolve(canvas);
        };

        img.onerror = function() {
            reject(new Error(`No ASL image found for letter "${letter}".`));
        };
    });
}


function mergeCanvases(leftCanvas, rightCanvas) {
    if (!leftCanvas) return rightCanvas;
    if (!rightCanvas) return leftCanvas;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = leftCanvas.width + rightCanvas.width;
    canvas.height = Math.max(leftCanvas.height, rightCanvas.height);

    ctx.drawImage(leftCanvas, 0, 0);
    ctx.drawImage(rightCanvas, leftCanvas.width, 0);

    return canvas;
}

document.getElementById("letter-input").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        const word = event.target.value;
        displayASLWordImage(word);
    }
});

document.getElementById("enter-button").addEventListener("click", function() {
    const word = document.getElementById("letter-input").value;
    displayASLWordImage(word);
});

document.getElementById("clear-button").addEventListener("click", function() {
    const letterImageContainer = document.getElementById("letter-image");
    const letterInput = document.getElementById("letter-input");

    letterInput.value = '';
    letterImageContainer.innerHTML = '';
});