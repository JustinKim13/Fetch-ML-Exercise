const form = document.getElementById('predict-form');
const resultDiv = document.getElementById('result');
const plotImage = document.getElementById('receipt-plot');

form.addEventListener('submit', function(event) {
    event.preventDefault();
    const month = document.getElementById('month').value;

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ month: parseInt(month) })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerText = data.error;
        } else {
            resultDiv.innerText = `Predicted Receipts for Month ${data.month}: ${data.predicted_receipts}`;
            plotImage.src = `/plot?month=${data.month}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
