window.onload = function() {
    setTimeout(function() {
        const messageField = document.getElementById('error_message');
        if (messageField) {
            messageField.classList.add('hidden');
        }
    }, 5000);
};

function syncColorDepthInputs(value) {
    if(value < 2){
        value = 2;
    }
    if(value > 255){
        value = 255;
    }
    if(value > 16){
        document.getElementById("color_depth_alert").textContent = "high values can cause very long loading times!";
    } else {
        document.getElementById("color_depth_alert").textContent = "";
    }
    document.getElementById("color_depth_sl").value = value;
    document.getElementById("color_depth_nr").value = value;
}

function showLoading(){
    document.getElementById("loading").style.display = "flex";
}

function hideLoading(){
    document.getElementById("loading").style.display = "none";
}

let mediaRecorder;
let audioChunks = [];

const recordButton = document.getElementById("record_button");
const stopButton = document.getElementById("stop_button");
const microphoneSelect = document.getElementById("microphone_select");

navigator.mediaDevices.enumerateDevices().then(devices => {
    devices.forEach(device => {
        if(device.kind === "audioinput"){
            const option = document.createElement("option");
            option.value = device.deviceId;
            option.textContent = device.label || `Microphone ${microphoneSelect.length}`;
            microphoneSelect.appendChild(option);
        }
    })
})

recordButton.addEventListener("click", async () => {
    const selected_device_id = microphoneSelect.value;
    const constraints = {
        audio: {
            deviceId: selected_device_id !== 'default' ? {exact: selected_device_id} : undefined,
            channelCount: 1,
            sampleSize: 16,
            sampleRate: 44100
        }
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints)
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
        showLoading()
        const audioBlob = new Blob(audioChunks, {type: 'audio/wav'});
        audioChunks = [];

        const formData = new FormData();
        formData.append('audio', audioBlob);

        fetch('/save_recording', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
            .then(data => {
                if (data.success) {
                    hideLoading();
                    window.location.href = `/recorded_audio?filename=${data.filename}`;
                } else {
                    alert('error saving recording: ' + data.error)
                }
            }).catch(error => {
                console.error('error saving recording:', error);
        })
    };
    
    mediaRecorder.start();
    recordButton.classList.add('blinking');
    recordButton.disabled = true;
    stopButton.disabled = false;
    stopButton.classList.add('enabled');
});

stopButton.addEventListener('click', () => {
    mediaRecorder.stop();
    recordButton.classList.remove('blinking');
    recordButton.disabled = false;
    stopButton.disabled = true;
    stopButton.classList.remove('enabled');
});

