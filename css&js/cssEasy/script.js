const letterImages = [
    'static/ASL/A.png',
    'static/ASL/B.png', 
    'static/ASL/C.png', 
    'static/ASL/D.png',
    'static/ASL/E.png',
    'static/ASL/L.png',
    'static/ASL/M.png',
    'static/ASL/N.png',
    'static/ASL/O.png',
    'static/ASL/S.png',
    'static/ASL/T.png',
];

const letters = "ABCDELMNOST".split("");

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
