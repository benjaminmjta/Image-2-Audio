let color_button = document.getElementById("color_switch");
let color_slider = document.getElementById("color_depth_sl");
let color_label = document.getElementById("color_depth_label");

window.onload = function() {
    color_slider.value = 1;
    ColorDepthControl();
    setTimeout(function() {
        const messageField = document.getElementById('error_message');
        if (messageField) {
            messageField.classList.add('hidden');
        }
    }, 5000);
};

color_button.addEventListener("click", SwitchColorGray);

function ButtonGray(){
    color_button.textContent = "converting in grayscale";
    color_button.classList.remove('rainbow');
}

function ButtonColor(){
    color_button.textContent = "converting in color";
    color_button.classList.add('rainbow');
}


function SwitchColorGray(){
    if(color_slider.value < 4){
        ButtonColor()
        color_slider.value = 4;
        ColorDepthControl(4);
    }
    else{
        ButtonGray()
        color_slider.value = 1;
        ColorDepthControl(1);
    }

}

function ColorDepthControl(){
    let value = color_slider.value
    switch (value) {
        case '1':
            color_label.textContent = "1 Bit";
            ButtonGray();
            break;
        case '2':
            color_label.textContent = "2 Bit";
            ButtonGray();
            break;
        case '3':
            color_label.textContent = "3 Bit";
            ButtonGray();
            break;
        case '4':
            color_label.textContent = "4 Bit";
            ButtonColor();
            break;
        case '5':
            color_label.textContent = "8 Bit";
            ButtonColor();
            break;
        default:
            break;
    }
    if(value > 4){
        document.getElementById("color_depth_alert").textContent = "color depth > 4 can cause very long loading times!";
    } else {
        document.getElementById("color_depth_alert").textContent = "";
    }
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

