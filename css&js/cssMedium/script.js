const letterImages = [
    'static/ASL/F.png',
    'static/ASL/G.png', 
    'static/ASL/H.png', 
    'static/ASL/I.png',
    'static/ASL/J.png',
    'static/ASL/K.png',
    'static/ASL/P.png',
    'static/ASL/Q.png',
    'static/ASL/R.png',
    'static/ASL/U.png',
    'static/ASL/V.png',
];


const letters = "FGHIJKPQRUV".split("");

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

