import React, { useState, useEffect } from 'react';
import './App.css';

// API URL configuration - works in both development and production
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [users, setUsers] = useState([]);
  const [formData, setFormData] = useState({
    user_id: '',
    requested_amount: '5000000',
    loan_purpose: 'business',
    contact_email: 'applicant@example.com',
    contact_phone: '+256700000000'
  });

  const [decision, setDecision] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API_URL}/users`)
      .then(res => res.json())
      .then(data => {
        setUsers(data.users || []);
        if (data.users && data.users.length > 0) {
          setFormData(prev => ({ ...prev, user_id: data.users[0] }));
        }
      })
      .catch(err => console.error('Failed to load users:', err));
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setDecision(null);

    try {
      const response = await fetch(
        `${API_URL}/evaluate?user_id=${formData.user_id}&requested_amount=${formData.requested_amount}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          }
        }
      );

      if (!response.ok) throw new Error('Failed to evaluate application');

      const data = await response.json();
      setDecision(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getDecisionClass = (decision) => {
    if (decision === 'auto_approve') return 'approved';
    if (decision === 'auto_reject') return 'rejected';
    return 'review';
  };

  const getDecisionLabel = (decision) => {
    if (decision === 'auto_approve') return 'APPROVED';
    if (decision === 'auto_reject') return 'REJECTED';
    return 'REQUIRES REVIEW';
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-UG', {
      style: 'currency',
      currency: 'UGX',
      minimumFractionDigits: 0
    }).format(amount);
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-container">
          <div className="bank-logo">Uganda Financial Services</div>
          <h1>Loan Eligibility Assessment System</h1>
          <p>Automated creditworthiness evaluation powered by machine learning</p>
        </div>
      </header>

      <main className="App-main">
        {!decision ? (
          <form onSubmit={handleSubmit} className="application-form">
            <div className="form-header">
              <h2>Loan Application</h2>
              <p className="subtitle">Complete the form below to receive an instant eligibility assessment</p>
            </div>

            <div className="form-grid">
              <div className="form-group">
                <label>Applicant ID</label>
                <select
                  name="user_id"
                  value={formData.user_id}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">Select applicant...</option>
                  {users.map(user => (
                    <option key={user} value={user}>{user}</option>
                  ))}
                </select>
                <small>Select from existing applicants in the system</small>
              </div>

              <div className="form-group">
                <label>Requested Amount (UGX)</label>
                <div className="input-prefix">
                  <input
                    type="number"
                    name="requested_amount"
                    value={formData.requested_amount}
                    onChange={handleInputChange}
                    required
                    min="1000000"
                    max="50000000"
                    step="100000"
                  />
                </div>
                <small>Minimum: UGX 1,000,000 | Maximum: UGX 50,000,000</small>
              </div>
            </div>

            <div className="form-grid">
              <div className="form-group">
                <label>Loan Purpose</label>
                <select
                  name="loan_purpose"
                  value={formData.loan_purpose}
                  onChange={handleInputChange}
                >
                  <option value="business">Business Expansion</option>
                  <option value="agriculture">Agriculture</option>
                  <option value="education">Education</option>
                  <option value="healthcare">Healthcare</option>
                  <option value="housing">Housing</option>
                  <option value="personal">Personal</option>
                </select>
              </div>

              <div className="form-group">
                <label>Contact Phone</label>
                <input
                  type="tel"
                  name="contact_phone"
                  value={formData.contact_phone}
                  onChange={handleInputChange}
                  required
                  placeholder="+256700000000"
                />
              </div>
            </div>

            <div className="form-grid full-width">
              <div className="form-group">
                <label>Email Address</label>
                <input
                  type="email"
                  name="contact_email"
                  value={formData.contact_email}
                  onChange={handleInputChange}
                  required
                  placeholder="applicant@example.com"
                />
              </div>
            </div>

            {error && <div className="error-message">{error}</div>}

            <button type="submit" className={`btn btn-primary ${loading ? 'loading' : ''}`} disabled={loading}>
              {loading ? 'Processing Application...' : 'Submit Application'}
            </button>
          </form>
        ) : (
          <div className="decision-result">
            <div className="decision-card">
              <div className={`decision-header ${getDecisionClass(decision.decision)}`}>
                <div className="decision-status">Application Status</div>
                <div className="decision-title">{getDecisionLabel(decision.decision)}</div>
                <div className="decision-subtitle">
                  Confidence Level: {(decision.confidence * 100).toFixed(1)}%
                </div>
              </div>

              <div className="decision-details">
                <div className="detail-section">
                  <h3>Application Details</h3>
                  <div className="detail-row">
                    <span className="label">Application Reference</span>
                    <span className="value">{decision.application_id.substring(0, 8).toUpperCase()}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Approval Probability</span>
                    <span className="value highlight">{(decision.probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Requested Amount</span>
                    <span className="value">{formatCurrency(decision.requested_amount)}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Recommended Amount</span>
                    <span className="value highlight">{formatCurrency(decision.recommended_amount)}</span>
                  </div>
                </div>

                {decision.explanation && (
                  <div className="financial-profile">
                    <h3>Financial Profile Summary</h3>
                    <div className="stats-grid">
                      <div className="stat-item">
                        <span className="stat-label">Total Income</span>
                        <span className="stat-value">
                          {formatCurrency(decision.explanation.income_total || 0)}
                        </span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">Total Expenses</span>
                        <span className="stat-value">
                          {formatCurrency(decision.explanation.expense_total || 0)}
                        </span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">Average Balance</span>
                        <span className="stat-value">
                          {formatCurrency(decision.explanation.balance_avg || 0)}
                        </span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">Transaction Count</span>
                        <span className="stat-value">
                          {decision.explanation.transaction_count || 0}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                <button onClick={() => setDecision(null)} className="btn btn-secondary">
                  New Application
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <div className="footer-content">
          <p>Uganda Financial Services - Loan Eligibility Assessment System</p>
          <p>Powered by machine learning creditworthiness analysis</p>
          <div className="api-status">
            <span className="status-dot"></span>
            System Operational
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
