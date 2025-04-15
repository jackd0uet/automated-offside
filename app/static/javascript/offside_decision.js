const csrftoken = Cookies.get('csrftoken');

const decisionText = document.getElementById("decisionText");
let finalDecision = null;

const overrideFormContainer = document.getElementById("overrideFormContainer");
const overrideForm = document.getElementById("overrideForm");

overrideForm.addEventListener("submit", function(event) {
    event.preventDefault();

    const clickedBtn = event.submitter;
    
    if (clickedBtn.id == "confirmDecisionBtn") {
        finalDecision = algorithmDecision;
    }
    else if (clickedBtn.id == "overrideDecisionBtn") {
        finalDecision = algorithmDecision == "Onside" ? "Offside" : "Onside";
    }

    decisionText.textContent = "Final decision: " + finalDecision;
    overrideFormContainer.style.display = 'none';

    fetch(storeOffsideUrl, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrftoken
        },
        body: JSON.stringify({
            algorithm_decision: algorithmDecision,
            final_decision: finalDecision
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Failed to save image");
        }
        return response.json();
    })
    .then(data => {
        decisionId = data.decision_id
        showAlert("Offside decision saved successfully!", "success");
    })
    .catch(error => {
        console.error("Error saving decision:", error)
    })
});