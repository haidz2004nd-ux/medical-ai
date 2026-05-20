import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // Đây là dòng quan trọng để nạp CSS
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);