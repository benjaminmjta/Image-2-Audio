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