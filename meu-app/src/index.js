import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import MeuComponente from './dashboard';
import Input from './input';
import Login from './login';
import Cadastro from './cadastro';
import Item from './item';
import './index.css';
import './style.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/dashboard" element={<MeuComponente />} />
        <Route path="/input" element={<Input />} />
        <Route path="/login" element={<Login />} />
        <Route path="/cadastro" element={<Cadastro />} />
        <Route path="/item" element={<Item />} />
        <Route path="/item/:id" element={<Item />} />
        <Route path="*" element={<MeuComponente />} />
      </Routes>
    </BrowserRouter>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);

export default App;
