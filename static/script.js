document.getElementById('summary-form').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the default form submission behavior

    const column = document.getElementById('column').value;  // Get the selected column name

    // Prepare the request data for summary statistics
    const requestData = {
        column: column             // Send the column name selected by the user
    };

    // Send the POST request to the Flask server
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)  // Convert the request data to JSON format
    })
    .then(response => response.json())  // Parse the response as JSON
    .then(data => {
        if (data.error) {
            // If there is an error in the response, display the error message
            console.error('Error:', data.error);
            alert(data.error);
        } else {
            // Display summary statistics
            let resultDiv = document.getElementById('summary-results');
            resultDiv.innerHTML = '';  // Clear any previous results

            for (let stat in data) {
                resultDiv.innerHTML += `<p><strong>${stat}:</strong> ${data[stat]}</p>`;
            }
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('There was an error processing your request.');
    });
});

// Separate logic for scatterplot (assuming you have scatterplot code here)
function renderScatterplot() {
    // Your scatterplot rendering code goes here
}
