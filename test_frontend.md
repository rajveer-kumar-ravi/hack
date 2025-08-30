# Frontend Functionality Test Guide

## 🧪 Testing Checklist

### 1. Model Status Display
- [ ] Open http://localhost:3001
- [ ] Check if model status shows "Ready" instead of "Idle"
- [ ] Verify status updates after batch analysis
- [ ] Check console for any API errors

### 2. Dashboard Functionality
- [ ] Verify all statistics cards display correctly
- [ ] Check navigation links work
- [ ] Verify quick action buttons function
- [ ] Test responsive design on different screen sizes

### 3. Batch Analysis
- [ ] Navigate to Batch Analysis page
- [ ] Test file upload with sample_data.csv
- [ ] Verify drag & drop functionality
- [ ] Test sample size input
- [ ] Check loading states during analysis
- [ ] Verify results display after analysis
- [ ] Test error handling with invalid files

### 4. Real-Time Analysis
- [ ] Navigate to Real-Time Analysis page
- [ ] Fill out transaction form with test data
- [ ] Submit analysis request
- [ ] Verify risk assessment display
- [ ] Check recommendation logic
- [ ] Test form validation

### 5. Results Page
- [ ] Navigate to Results page after batch analysis
- [ ] Verify charts and visualizations
- [ ] Check flagged transactions table
- [ ] Test export functionality
- [ ] Verify statistics display

### 6. API Communication
- [ ] Check browser console for API calls
- [ ] Verify CORS headers
- [ ] Test error handling
- [ ] Check loading states

## 🔧 Debugging Steps

### If Model Status Shows "Idle":
1. Open browser developer tools (F12)
2. Check Console tab for errors
3. Check Network tab for API calls
4. Verify backend is running on port 5000
5. Test API directly: `curl http://localhost:5000/api/model-status`

### If File Upload Fails:
1. Check file format (must be CSV)
2. Verify file size (max 2GB)
3. Check browser console for errors
4. Verify backend uploads folder exists

### If Real-Time Analysis Fails:
1. Ensure batch analysis was run first
2. Check form validation
3. Verify all required fields are filled
4. Check API response in Network tab

## 📊 Expected Results

### After Batch Analysis with sample_data.csv:
- Total Transactions: 10
- Fraudulent Detected: 5
- Legitimate Transactions: 5
- Model Accuracy: ~90%+
- Model Status: "Ready"

### Real-Time Analysis Test Data:
```json
{
  "Transaction_ID": "TEST001",
  "User_ID": "TESTUSER",
  "Transaction_Amount": 5000,
  "Transaction_Type": "TRANSFER",
  "Merchant_Category": "OTHER"
}
```

## 🚨 Common Issues & Solutions

### Issue: Model Status Always "Idle"
**Solution**: Check API communication and CORS configuration

### Issue: File Upload Not Working
**Solution**: Verify file format and backend uploads directory

### Issue: Real-Time Analysis Returns Error
**Solution**: Run batch analysis first to train the model

### Issue: Charts Not Displaying
**Solution**: Check if Recharts library is properly installed

## ✅ Success Criteria

- [ ] All pages load without errors
- [ ] Model status updates correctly
- [ ] File upload works with sample data
- [ ] Real-time analysis provides risk assessment
- [ ] Results page displays charts and tables
- [ ] Navigation works between all pages
- [ ] Responsive design works on mobile
- [ ] No console errors
- [ ] All API calls return 200 status
