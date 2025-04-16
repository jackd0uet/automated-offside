const csrftoken = Cookies.get('csrftoken');

const filePickerContainer = document.getElementById("filePickerContainer");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const imageContainer = document.getElementById("imageContainer");

const progressContainer = document.getElementById("progressContainer");
const progressBar = document.getElementById("progressBar");

const detectionContainer = document.getElementById("detectionContainer");
const resultImage = document.getElementById("resultImage");

const uploadForm = document.getElementById("uploadForm");
const runDetectionBtn = document.getElementById("runDetectionBtn");

const adjustmentsMenu = document.getElementById("adjustmentsMenu");
const adjustmentsForm = document.getElementById("adjustmentsForm");
const tweaksForm = document.getElementById("tweaksForm");

const confidenceSlider = document.getElementById("confidenceSlider");
const confidenceValue = document.getElementById("confidenceValue");

const defendingForm = document.getElementById("defendingForm");

const adjustmentsOffCanvas = new bootstrap.Offcanvas(adjustmentsMenu, {
            backdrop: false
        });

let detectionData = null;
let defending_team = null;

confidenceSlider.addEventListener("input", function(event) {
    confidenceValue.textContent = event.target.value;
});

// TODO: fix progress bar
function startProgressBar() {
    // Show progress bar
    progressBar.style.width = "0%";
    progressBar.textContent = "0%";
    progressContainer.style.display = "block";
    progressContainer.classList.add("show");

    let progress = 0;
    
    progressInterval = setInterval(() => {
        let width = parseInt(progressBar.style.width);
        if (width < 95) {
            width += 5
            progressBar.style.width = width + "%";
            progressBar.textContent = width + "%";
        }
    }, 500);
}

function stopProgressBar() {
    clearInterval(progressInterval);
    progressBar.style.width = "100%";
    progressBar.textContent = "100%";

    progressContainer.style.display = "none";
    progressContainer.classList.remove("show");
}

function runImageDetection(formData, overwriteDetection) {
    startProgressBar();
    runDetectionBtn.disabled = true;
    runDetectionBtn.textContent = "Detection in progress...";

    let detectionPromise;

    if (overwriteDetection) {
        detectionPromise = fetch(processImageUrl, {
            method: "POST",
            headers: {
                "X-CSRFToken": csrftoken
            },
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to process image");
            }
            return response.json();
        })
        .then(data => {
            detectionData = data;
            // Send detections to render the pitch
            return renderDetection();
        });
        } else {
            detectionPromise = renderDetection();
        }

        detectionPromise
            .catch(error => {
                console.error("Error:", error);
                alert("Error processing image. Please try again.");

                filePickerContainer.style.display = "block";
                runDetectionBtn.disabled = false;
                runDetectionBtn.textContent = "Run Detection";
            })
            .finally(() => {
                stopProgressBar();
                
            });
}

function renderDetection() {
    return fetch(renderPitchUrl, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrftoken
        },
        body: JSON.stringify({
            ball_xy: detectionData.ball_xy,
            players_xy: detectionData.players_xy,
            refs_xy: detectionData.refs_xy,
            players_detections: detectionData.players_detections
        })
    })
    .then(response => response.blob())
    .then(blob => {
        const imgUrl = URL.createObjectURL(blob);
        resultImage.src = imgUrl;

        runDetectionBtn.style.display = "none";
        detectionContainer.style.display = "block";
        detectionContainer.classList.add("show");

        adjustmentsOffCanvas.show();
    });
}

function generatePlayerTweaksForm(players) {
    const form = document.createElement("form");
    form.id = "dynamicTweaksForm";
    form.classList.add("mb-3");

    players.tracker_id.forEach((id, index) => {
        const classId = players.class_id[index];
        const className = players.class_name[index];

        const playerDiv = document.createElement("div");
        playerDiv.classList.add("mb-3");

        playerDiv.innerHTML = `
            <label class="form-label d-block">
                Player ID: <strong>${id}</strong> (${className})
            </label>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="radio" name="team_${id}" id="teamA_${id}" value="0" ${classId === 0 ? "checked" : ""}>
                <label class="form-check-label" for="teamA_${id}">Team A</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="radio" name="team_${id}" id="teamB_${id}" value="1" ${classId === 1 ? "checked" : ""}>
                <label class="form-check-label" for="teamB_${id}">Team B</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="delete_${id}" name="delete_${id}">
                <label class="form-check-label text-danger" for="delete_${id}">Delete</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="goalkeeper_${id}" name="goalkeeper_${id}" ${className === "goalkeeper" ? "checked" : ""}>
                <label class="form-check-label text-warning" for="goalkeeper_${id}">Goalkeeper</label>
            </div>
            <hr/>
        `;
        form.appendChild(playerDiv);
    });

    const confirmBtn = document.createElement("button");
    confirmBtn.type = "submit";
    confirmBtn.classList.add("btn", "btn-primary", "mt-2");
    confirmBtn.textContent = "Apply Changes";

    form.appendChild(confirmBtn);

    return form;
}

function waitForDefendingAssignment() {
    return new Promise((resolve) => {
        adjustmentsForm.style.display = "none";
        defendingForm.style.display = "block";

        const isTeamADefending = document.getElementById("teamA").checked;

        const handler = function(event) {
            event.preventDefault();

            defending_team = isTeamADefending ? 0 : 1;
            defendingForm.removeEventListener("submit", handler);

            resolve();
        };

        defendingForm.addEventListener("submit", handler)
    });
}

imageInput.addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;

            imageContainer.style.display = "block";
            imageContainer.classList.add("show")                
        };
        reader.readAsDataURL(file);
    }
});

uploadForm.addEventListener("submit", function(event) {
    event.preventDefault();

    if (imageInput.files.length === 0) {
        alert("Please select an image.");
        return;
    }

    // Hide file picker
    filePickerContainer.style.display = "none";
    
    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    runImageDetection(formData, true);
});

adjustmentsForm.addEventListener("submit", async function(event) {
    event.preventDefault();

    const clickedBtn = event.submitter;

    if (clickedBtn.id === "confirmOffsideBtn") {

        if (!detectionData.players_detections.class_name.includes("goalkeeper")) {
            await waitForDefendingAssignment();
        }

        fetch(classifyOffsideUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrftoken
            },
            body: JSON.stringify({
                detection_data: detectionData,
                defending_team: defending_team
            })
        })
        .then(response => response.json())
        .then(data => {
            window.location.href = data.redirect_url;
        })
        .catch(error => {
            console.error("Error:", error);
            alert("Error");
        });
    }
    else if (clickedBtn.id === "cancelDetectionBtn") {
        location.reload();
    }
    else if (clickedBtn.id === "rerunDetectionBtn") {
        const confidenceValue = parseFloat(confidenceSlider.value);
        if (confidenceValue == 0.5) {
            alert("Please pick a non-default value if you would like to run new detections, otherwise continue to Offside Detection.");
            return;
        }
        else {
            adjustmentsOffCanvas.hide();
            detectionContainer.style.display = "none";
            detectionContainer.classList.remove("show");
            runDetectionBtn.style.display = "inline-block";

            const formData = new FormData();
            formData.append("image", imageInput.files[0]);
            formData.append("confidence", confidenceSlider.value.toString());

            runImageDetection(formData, true);
        }
    }
    else if (clickedBtn.id == "manualTeamChange") {
        adjustmentsForm.style.display = "none";
        const dynamicForm = generatePlayerTweaksForm(detectionData.players_detections);
        tweaksForm.style.display = "block";
        tweaksForm.appendChild(dynamicForm);

        dynamicForm.addEventListener("submit", function(event) {
            event.preventDefault();

            const updatedDetections = JSON.parse(JSON.stringify(detectionData.players_detections));
            const updatedPlayersXY = JSON.parse(JSON.stringify(detectionData.players_xy));

            for (let i = updatedDetections.tracker_id.length - 1; i >= 0; i--) {
                const id = updatedDetections.tracker_id[i];
                const teamValue = dynamicForm.querySelector(`input[name="team_${id}"]:checked`).value;
                const deleteChecked = dynamicForm.querySelector(`#delete_${id}`).checked;
                const goalkeeperChecked = dynamicForm.querySelector(`#goalkeeper_${id}`).checked;

                if (goalkeeperChecked) {
                    updatedDetections['class_name'][i] = 'goalkeeper'
                } else {
                    updatedDetections['class_name'][i] = 'player'
                }

                if (deleteChecked) {
                    ['tracker_id', 'class_id', 'class_name', 'confidence'].forEach(field => {
                        updatedDetections[field].splice(i, 1);
                    });
                    // TODO: confirm this is working
                    updatedPlayersXY.xy.splice(i, 1);
                    updatedPlayersXY.tracker_id.splice(i, 1);
                } else {
                    updatedDetections.class_id[i] = parseInt(teamValue);
                }
            }

            detectionData.players_detections = updatedDetections;
            detectionData.players_xy = updatedPlayersXY;

            fetch(updateDetectionsUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRFToken": csrftoken
                },
                body: JSON.stringify(detectionData)
            })
            .then(response => response.json());
            adjustmentsForm.style.display = "block";
            tweaksForm.style.display = "none";
            tweaksForm.removeChild(dynamicForm);

            showAlert("Object detection updated successfully!", "success")

            adjustmentsOffCanvas.hide();
            runImageDetection(null, false);
        });
    }
});