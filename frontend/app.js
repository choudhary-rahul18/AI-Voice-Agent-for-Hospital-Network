// frontend/app.js
let mediaRecorder;
let audioChunks = [];
const recordBtn = document.getElementById('recordBtn');
const statusText = document.getElementById('status');
const audioPlayback = document.getElementById('audioPlayback');

recordBtn.onclick = async () => {
    // If currently recording, STOP it and send to backend
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordBtn.textContent = "Start Recording";
        statusText.textContent = "Status: Thinking... (Sending to Backend)";
    } 
    // If idle, START recording
    else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                // Bundle the audio chunks into a Blob
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioChunks = []; // Reset for next time
                
                // Prepare form data to send to FastAPI
                const formData = new FormData();
                formData.append("audio_file", audioBlob, "recording.wav");
                formData.append("session_id", "session_123"); // Hardcoded for POC session tracking

                try {
                    const response = await fetch("http://localhost:8000/chat", {
                        method: "POST",
                        body: formData
                    });
                    
                    if (response.ok) {
                        // Receive the audio file from FastAPI and play it
                        const audioBlobResponse = await response.blob();
                        const audioUrl = URL.createObjectURL(audioBlobResponse);
                        audioPlayback.src = audioUrl;
                        audioPlayback.style.display = "block"; // Show the player
                        audioPlayback.play();
                        statusText.textContent = "Status: Ready";
                    } else {
                        statusText.textContent = "Status: Error on backend";
                    }
                } catch (error) {
                    console.error(error);
                    statusText.textContent = "Status: Network Error";
                }
            };
            
            mediaRecorder.start();
            recordBtn.textContent = "Stop Recording";
            statusText.textContent = "Status: Recording... Speak now.";
            audioPlayback.style.display = "none"; // Hide player while recording
        } catch (err) {
            statusText.textContent = "Error: Please allow microphone access.";
        }
    }
};