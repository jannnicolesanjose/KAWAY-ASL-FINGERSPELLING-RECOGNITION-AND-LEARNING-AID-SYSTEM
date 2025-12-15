const letterImages = [
    'static/ASL/W.png',
    'static/ASL/X.png', 
    'static/ASL/Y.png', 
    'static/ASL/Z.png',
];

const letters = "WXYZ".split("");

let currentIndex = 0;

const currentLetterImage = document.getElementById("current-letter-image");
const currentLetter = document.getElementById("current-letter");
const prevButton = document.getElementById("prev-button");
const nextButton = document.getElementById("next-button");

function updateImage() {
    currentLetterImage.src = letterImages[currentIndex];
    currentLetter.textContent = letters[currentIndex];
}

prevButton.addEventListener("click", () => {
    if (currentIndex > 0) {
        currentIndex--;
        updateImage();
    }
});

nextButton.addEventListener("click", () => {
    if (currentIndex < letterImages.length - 1) {
        currentIndex++;
        updateImage();
    }
});

updateImage();
