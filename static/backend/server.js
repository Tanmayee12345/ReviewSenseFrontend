const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
app.use(cors()); // Enable CORS
app.use(express.json());

const csvFilePath = path.join(__dirname, 'updated_reviews (1).csv');

app.post('/api/addReview', (req, res) => {
  const { clothing_id, age, title, review_text, rating, positive_feedback_count, division_name, department_name, class_name } = req.body;

  const newRow = `${clothing_id},${age},"${title}","${review_text}",${rating},1,${positive_feedback_count},"${division_name}","${department_name}","${class_name}"\n`;

  fs.appendFile(csvFilePath, newRow, (err) => {
    if (err) {
      console.error('Error writing to CSV:', err);
      return res.status(500).send('Error saving review.');
    }
    res.send('Review added successfully!');
  });
});

app.listen(3000, () => console.log('Server running on http://localhost:3000'));
