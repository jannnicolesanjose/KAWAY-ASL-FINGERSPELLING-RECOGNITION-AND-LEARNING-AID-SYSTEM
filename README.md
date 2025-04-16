# KAWAY-ASL-FINGERSPELLING-RECOGNITION-AND-LEARNING-AID-SYSTEM
KAWAY: American Sign Language (ASL) Fingerspelling Recognition and Learning Aid System

KAWAY is a web-based application designed to recognize and translate American Sign Language (ASL) fingerspelling using computer vision and machine learning. 
The system aims to support the Deaf and Hard-of-Hearing (HoH) community, ASL learners, educators, and anyone interested in inclusive communication.
Using a webcam and the MediaPipe framework, the app detects hand landmarks and processes them with a Random Forest model trained on a custom dataset of over 23,000 images. 
The application is capable of translating static ASL hand signs into text in real-time, providing users immediate feedback to improve their signing accuracy and consistency.

KAWAY features four main modules:
Learn: Offers instructional content, including videos and images, to help users learn the ASL alphabet.
Practice: Allows users to test their fingerspelling skills by replicating signs in front of their camera, with difficulty levels ranging from easy to hard.
Translate: Provides ASL-to-text and text-to-ASL conversion. Users can either sign letters to generate text or type words to see their corresponding ASL representations.
About Us: Shares information about the project and its developers.

Developed using Python (OpenCV, MediaPipe, Scikit-learn), HTML, CSS, and JavaScript, KAWAY was built with accessibility, accuracy, and educational value in mind. 
It uses the Random Forest algorithm for its ability to handle noisy data and deliver fast and reliable classification results, even under varying lighting and background conditions.
KAWAY is a step toward bridging communication gaps and promoting sign language learning through technology.
The project reflects a commitment to inclusivity and serves as a valuable resource for both self-learners and educators.

