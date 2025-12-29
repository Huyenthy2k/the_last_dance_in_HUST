/**
 * MAFIA Training Dashboard - Real-time Updates
 *
 * Polls the server every 500ms to fetch the latest DisplayState
 * and updates the DOM accordingly.
 */

// Configuration
const POLL_INTERVAL = 500;  // ms
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 2000;  // ms

// State
let reconnectAttempts = 0;
let isConnected = false;
let lastUpdateTime = null;

// ═══════════════════════════════════════════════════════════════
// FORMAT HELPERS
// ═══════════════════════════════════════════════════════════════

function formatNumber(n, decimals = 2) {
    if (n === null || n === undefined || isNaN(n)) return 'N/A';
    return n.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function formatPercent(n, showSign = true) {
    if (n === null || n === undefined || isNaN(n)) return 'N/A';
    const sign = showSign && n >= 0 ? '+' : '';
    return sign + formatNumber(n) + '%';
}

function formatMoney(n) {
    if (n === null || n === undefined || isNaN(n)) return 'N/A';
    return '$' + n.toLocaleString('en-US', { maximumFractionDigits: 0 });
}

function formatTime(seconds) {
    if (seconds === null || seconds === undefined || isNaN(seconds)) return '--';
    if (seconds < 60) return Math.round(seconds) + 's';
    if (seconds < 3600) {
        const m = Math.floor(seconds / 60);
        const s = Math.round(seconds % 60);
        return m + 'm ' + s + 's';
    }
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    return h + 'h ' + m + 'm';
}

function getColorClass(value, threshold = 0) {
    if (value === null || value === undefined) return 'dim';
    return value >= threshold ? 'green' : 'red';
}

// ═══════════════════════════════════════════════════════════════
// UPDATE FUNCTIONS
// ═══════════════════════════════════════════════════════════════

function updateDashboard(state) {
    if (!state || Object.keys(state).length === 0) {
        return;
    }

    // Update connection status
    updateConnectionStatus(true);

    // Header
    updateHeader(state);

    // Pipeline (2-Phase Overview)
    updatePipelineSection(state);

    // Data Splits (Train/Valid/Infer)
    updateDataSplitsSection(state);

    // Progress
    updateProgressSection(state);

    // Market
    updateMarketSection(state);

    // Portfolio
    updatePortfolioSection(state);

    // Observer
    updateObserverSection(state);

    // TD3
    updateTD3Section(state);

    // CBF
    updateCBFSection(state);

    // Performance
    updatePerformanceSection(state);

    // Events
    updateEventLog(state.event_log || []);
}

function updateHeader(state) {
    const phaseTitle = document.getElementById('phase-title');
    const phaseIcon = document.getElementById('phase-icon');

    if (state.training_mode === 'OBSERVER_ONLY') {
        phaseTitle.textContent = 'PHASE 1: OBSERVER TRAINING';
        phaseIcon.textContent = '🔮';
    } else if (state.training_mode === 'RL_ONLY') {
        phaseTitle.textContent = 'PHASE 2: TD3 TRAINING';
        phaseIcon.textContent = '🎯';
    } else {
        phaseTitle.textContent = 'MAFIA TRAINING';
        phaseIcon.textContent = '📊';
    }

    setText('seed', state.seed || 2025);
    setText('window', state.window || 0);

    // Run ID (could be date-based identifier)
    if (state.date) {
        setText('run-id', state.date.split(' ')[0] || '--');
    }
}

function updatePipelineSection(state) {
    const phase1 = document.getElementById('phase1-indicator');
    const phase2 = document.getElementById('phase2-indicator');
    const phase1Status = document.getElementById('phase1-status');
    const phase2Status = document.getElementById('phase2-status');
    const phase1Detail = document.getElementById('phase1-detail');
    const phase2Detail = document.getElementById('phase2-detail');
    const statusText = document.getElementById('status-text');

    const isPhase1 = state.training_mode === 'OBSERVER_ONLY';
    const currentPhase = state.phase || 'INIT';
    const warmupTarget = state.obs_warmup_target || 300;
    const warmupSamples = state.obs_warmup_samples || 0;
    const inObserverWarmup = isPhase1 && !state.obs_warmup_complete && warmupSamples < warmupTarget;

    if (isPhase1) {
        // Phase 1 Active
        phase1.className = 'phase-indicator active';
        phase2.className = 'phase-indicator pending';
        phase1Status.textContent = '● ACTIVE';
        phase1Status.className = 'phase-status active';
        phase2Status.textContent = '⏳ PENDING';
        phase2Status.className = 'phase-status pending';

        // Phase 1 details
        const iter = state.walkforward_iteration || 0;
        const total = state.walkforward_total_iterations || 5;
        const trainYears = state.walkforward_train_years || '--';
        const validYear = state.walkforward_valid_year || '--';
        phase1Detail.textContent = `Iter ${iter + 1}/${total} | ${trainYears} → ${validYear}`;
        phase2Detail.textContent = 'Will train after Observer completes';

        // Status bar
        if (inObserverWarmup) {
            statusText.textContent = `🔮 Observer Warmup: ${warmupSamples}/${warmupTarget} samples`;
        } else if (currentPhase === 'VALID') {
            statusText.textContent = `🔮 Validating Observer | Year: ${validYear}`;
        } else {
            statusText.textContent = `🔮 Observer Training | Epoch ${state.epoch || 0}/${state.total_epochs || 50}`;
        }
    } else {
        // Phase 2 Active (or Phase 1 completed)
        phase1.className = 'phase-indicator completed';
        phase2.className = 'phase-indicator active';
        phase1Status.textContent = '✓ DONE';
        phase1Status.className = 'phase-status completed';
        phase2Status.textContent = '● ACTIVE';
        phase2Status.className = 'phase-status active';

        // Phase details
        phase1Detail.textContent = 'Observer Frozen (Static Expert)';
        const bufferPct = state.td3_buffer_max > 0 ? Math.round(state.td3_buffer_size / state.td3_buffer_max * 100) : 0;
        phase2Detail.textContent = `Buffer: ${bufferPct}% | Updates: ${state.td3_updates || 0}`;

        // Status bar
        if (currentPhase === 'WARMUP') {
            statusText.textContent = `🎯 TD3 Warmup: ${state.td3_buffer_size || 0}/${state.td3_buffer_max || 0} buffer`;
        } else if (currentPhase.includes('TRAIN')) {
            statusText.textContent = `🎯 TD3 Training | Epoch ${state.epoch || 0}/${state.total_epochs || 50}`;
        } else {
            statusText.textContent = `🎯 Phase 2: ${currentPhase}`;
        }
    }
}

function updateDataSplitsSection(state) {
    // Show walk-forward data splits (full date ranges for Expanding Window)
    // Use full ranges if available, fallback to year-only (legacy)
    const trainRange = state.walkforward_train_range || state.walkforward_train_years || '--';
    const validRange = state.walkforward_valid_range || (state.walkforward_valid_year ? String(state.walkforward_valid_year) : '--');
    const inferRange = state.walkforward_infer_range || (state.walkforward_infer_year ? String(state.walkforward_infer_year) : '--');
    const isFinetune = state.walkforward_is_finetune || false;

    setText('train-range', trainRange);
    setText('valid-range', validRange);
    setText('infer-range', inferRange);

    // Show/hide finetune badge
    const badge = document.getElementById('finetune-badge');
    if (badge) {
        if (isFinetune) {
            badge.classList.remove('hidden');
            badge.textContent = '[Finetune]';
            badge.style.color = '#ffa500';  // Orange for finetune
        } else {
            badge.classList.remove('hidden');
            badge.textContent = '[Scratch]';
            badge.style.color = '#00ff00';  // Green for scratch
        }
    }
}

function updateProgressSection(state) {
    setText('epoch', state.epoch || 0);
    setText('total-epochs', state.total_epochs || 50);
    setText('step', state.step || 0);
    setText('total-steps', state.total_steps || 252);
    setText('date', state.date || '--');

    // Progress bar
    const progress = state.total_steps > 0 ? (state.step / state.total_steps) * 100 : 0;
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
        progressBar.style.width = Math.min(progress, 100) + '%';
    }

    // Timing
    setText('speed', formatNumber(state.speed, 1));
    setText('elapsed', formatTime(state.elapsed_seconds || 0));

    // Calculate ETA
    if (state.speed > 0 && state.total_steps > state.step) {
        const remainingSteps = state.total_steps - state.step;
        const eta = remainingSteps / state.speed;
        setText('eta', formatTime(eta));
    } else {
        setText('eta', '--');
    }
}

function updateMarketSection(state) {
    const dirEl = document.getElementById('direction');
    // Direction is numeric: 0=BEAR, 1=FLAT, 2=BULL
    const dirNum = state.direction !== undefined ? state.direction : 1;
    const dirMap = {0: ['🐻', 'BEAR'], 1: ['➡️', 'FLAT'], 2: ['🐂', 'BULL']};
    const [icon, dirText] = dirMap[dirNum] || dirMap[1];
    dirEl.textContent = icon + ' ' + dirText;
    dirEl.className = dirText.toLowerCase();

    setText('trend-z', formatNumber(state.trend_z, 2));
    setText('eta-risk', formatNumber(state.eta_risk, 2));
    setText('volatility', formatPercent(state.volatility * 100, true));
    setText('regime-shifts', state.regime_shift_count || 0);

    // Last regime shift
    if (state.last_regime_date && state.last_regime_from && state.last_regime_to) {
        setText('last-regime', `${state.last_regime_from}→${state.last_regime_to}`);
    } else {
        setText('last-regime', '--');
    }
}

function updatePortfolioSection(state) {
    const status = state.is_rebalance ? '🔄 REBALANCE' : '💤 HOLD';
    setText('rebalance-status', status);
    setText('rebalance-count', state.rebalance_count || 0);
    setText('top-k', state.top_k || 10);

    const daysSince = state.days_since_rebal || 0;
    const interval = state.rebal_interval || 14;
    setText('next-rebal', `${daysSince}/${interval} days`);

    setTextWithClass('n-added', state.n_added || 0, 'green');
    setTextWithClass('n-removed', state.n_removed || 0, 'red');
    setText('n-kept', state.n_kept || 0);
}

function updateObserverSection(state) {
    const content = document.getElementById('obs-content');
    const status = document.getElementById('obs-status');
    const section = document.getElementById('observer-section');
    const warmupTarget = state.obs_warmup_target || 300;
    const warmupSamples = state.obs_warmup_samples || 0;
    const inWarmup = !state.obs_warmup_complete && warmupSamples < warmupTarget;

    if (state.training_mode === 'OBSERVER_ONLY') {
        // Phase 1: Observer is active
        section.className = '';  // Reset any completed styles

        // Check warmup phase
        if (inWarmup) {
            // Warmup phase
            status.textContent = '⏳ WARMUP';
            status.className = 'status-badge warmup';

            const pct = warmupTarget > 0 ?
                (warmupSamples / warmupTarget * 100) : 0;
            const remaining = warmupTarget - warmupSamples;

            content.innerHTML = `
                <div class="metric-row">
                    <span class="yellow">⏳ WARMUP PHASE - Collecting samples before training</span>
                </div>
                <div class="metric-row">
                    <span>Buffer: ${warmupSamples}/${warmupTarget} (${formatNumber(pct, 1)}%)</span>
                    <span>Remaining: ${remaining} samples</span>
                </div>
                <div class="metric-row">
                    <div class="warmup-bar-container">
                        <div class="warmup-bar" style="width: ${pct}%"></div>
                    </div>
                </div>
            `;
        } else {
            // Training phase
            status.textContent = '● TRAINING';
            status.className = 'status-badge training';

            const dirClass = (state.obs_dir_pred || '').toLowerCase();
            const pgColor = getColorClass(state.obs_r_sel_total);

            content.innerHTML = `
                <div class="metric-row">
                    <span>Direction: <span class="${dirClass}">${state.obs_dir_pred || 'N/A'}</span></span>
                    <span>Conf: ${formatNumber((state.obs_dir_conf || 0) * 100, 1)}%</span>
                    <span>η: ${formatNumber(state.obs_eta_pred, 2)}</span>
                </div>
                <div class="metric-row">
                    <span>Loss Total: ${formatNumber(state.obs_loss_total, 4)}</span>
                    <span>Dir: ${formatNumber(state.obs_loss_dir, 4)}</span>
                    <span>Eta: ${formatNumber(state.obs_loss_eta, 4)}</span>
                    <span>PG: ${formatNumber(state.obs_loss_pg, 4)}</span>
                </div>
                <div class="metric-row">
                    <span>Samples Dir: ${state.obs_samples_dir || 0}</span>
                    <span>Risk: ${state.obs_samples_risk || 0}</span>
                    <span>Sel: ${state.obs_samples_sel || 0}</span>
                </div>
                <div class="metric-row">
                    <span>PG Reward: <span class="${pgColor}">${formatNumber(state.obs_r_sel_total, 4)}</span></span>
                    <span>mean_ret: ${formatNumber(state.obs_r_sel_return, 4)}</span>
                    <span>turnover: ${formatNumber(-(state.obs_r_sel_turnover || 0), 4)}</span>
                </div>
                <div class="metric-row">
                    <span>Best Score: ${formatNumber(state.walkforward_best_score, 4)} (Epoch ${state.walkforward_best_epoch || 0})</span>
                    <span>Current: ${formatNumber(state.walkforward_current_score, 4)}</span>
                </div>
            `;
        }
    } else {
        // RL_ONLY mode - Observer completed Phase 1 and is now frozen
        const section = document.getElementById('observer-section');
        section.className = 'section-completed';  // Mark as completed

        status.textContent = '✓ FROZEN';
        status.className = 'status-badge completed';

        const source = state.observer_checkpoint_source || 'N/A';
        const shortSource = source.length > 50 ? '...' + source.slice(-47) : source;
        const dirClass = (state.obs_dir_pred || '').toLowerCase();

        content.innerHTML = `
            <div class="metric-row completed-info">
                <span>✓ Phase 1 Complete - Acting as Static Expert</span>
            </div>
            <div class="metric-row">
                <span>Direction: <span class="${dirClass}">${state.obs_dir_pred || 'N/A'}</span></span>
                <span>Conf: ${formatNumber((state.obs_dir_conf || 0) * 100, 1)}%</span>
                <span>η: ${formatNumber(state.obs_eta_pred, 2)}</span>
            </div>
            <div class="metric-row dim">
                <span>Checkpoint: ${shortSource}</span>
            </div>
        `;
    }
}

function updateTD3Section(state) {
    const content = document.getElementById('td3-content');
    const status = document.getElementById('td3-status');
    const section = document.getElementById('td3-section');

    if (state.training_mode === 'RL_ONLY') {
        // Phase 2: TD3 is active
        section.className = '';  // Remove any dimming
        status.textContent = '● TRAINING';
        status.className = 'status-badge training';

        const bufferPct = state.td3_buffer_max > 0 ?
            (state.td3_buffer_size / state.td3_buffer_max * 100) : 0;
        const rewardColor = getColorClass(state.td3_reward);

        content.innerHTML = `
            <div class="metric-row">
                <span>Buffer: ${(state.td3_buffer_size || 0).toLocaleString()}/${(state.td3_buffer_max || 0).toLocaleString()} (${formatNumber(bufferPct, 1)}%)</span>
                <span>Updates: ${(state.td3_updates || 0).toLocaleString()}</span>
            </div>
            <div class="metric-row">
                <span>Actor Loss: ${formatNumber(state.td3_actor_loss, 6)}</span>
                <span>Critic Loss: ${formatNumber(state.td3_critic_loss, 6)}</span>
            </div>
            <div class="metric-row">
                <span>Reward: <span class="${rewardColor}">${formatNumber(state.td3_reward, 4)}</span></span>
                <span>Mean Q: ${formatNumber(state.td3_mean_q, 2)}</span>
            </div>
            <div class="metric-row">
                <span>R_return: ${formatNumber(state.td3_r_return, 4)}</span>
                <span>R_js: ${formatNumber(state.td3_r_js, 4)}</span>
                <span>JS_div: ${formatNumber(state.td3_js_divergence, 4)}</span>
            </div>
        `;
    } else {
        // Phase 1: TD3 is pending (waiting for Observer to complete)
        section.className = 'section-pending';  // Dim the section
        status.textContent = '⏳ PENDING';
        status.className = 'status-badge pending';

        content.innerHTML = `
            <div class="metric-row pending-info">
                <span>⏳ Waiting for Phase 2 - Observer training must complete first</span>
            </div>
            <div class="metric-row dim">
                <span>Current Action: Uniform weights [1/K, 1/K, ..., 1/K]</span>
            </div>
            <div class="metric-row dim">
                <span>Replay Buffer: Empty (will fill in Phase 2)</span>
            </div>
            <div class="metric-row dim">
                <span>Note: TD3 gradient updates are FROZEN during Observer training</span>
            </div>
        `;
    }
}

function updateCBFSection(state) {
    const cbfEnabled = state.cbf_enabled !== false;
    const status = document.getElementById('cbf-status');
    const trainingMode = state.training_mode || 'RL_ONLY';

    if (cbfEnabled) {
        status.textContent = '● ENABLED';
        status.className = 'status-badge enabled';
    } else if (trainingMode === 'OBSERVER_ONLY') {
        // Phase 1: TD3 frozen, CBF not applicable (uniform weights)
        status.textContent = '⏳ PENDING';
        status.className = 'status-badge pending';
    } else {
        status.textContent = '○ DISABLED';
        status.className = 'status-badge frozen';
    }

    setText('cbf-alpha', formatNumber(state.cbf_alpha, 4));
    setText('cbf-safety', formatNumber(state.cbf_safety_margin, 4));
    setText('cbf-max-pos', formatNumber(state.cbf_max_position, 2));

    const sigmaBase = formatNumber(state.cbf_sigma_base, 2);
    const sigmaCurr = formatNumber(state.cbf_sigma_current, 2);
    setText('cbf-sigma', `${sigmaBase} → ${sigmaCurr}`);
    setText('cbf-interventions', state.cbf_interventions || 0);
}

function updatePerformanceSection(state) {
    setText('capital', formatNumber(state.capital, 0));

    const dailyEl = document.getElementById('daily-return');
    if (dailyEl) {
        dailyEl.textContent = formatPercent(state.daily_return * 100);
        dailyEl.className = getColorClass(state.daily_return);
    }

    const cumulEl = document.getElementById('cumul-return');
    if (cumulEl) {
        cumulEl.textContent = formatPercent(state.cumul_return * 100);
        cumulEl.className = getColorClass(state.cumul_return);
    }

    setText('sharpe', formatNumber(state.sharpe, 2));

    const mddEl = document.getElementById('mdd');
    if (mddEl) {
        // BUG FIX: state.mdd is already in percentage (from callback), don't multiply by 100 again
        mddEl.textContent = formatPercent(state.mdd, false);
        mddEl.className = 'red';  // MDD is always negative indicator
    }

    setText('win-rate', formatNumber((state.win_rate || 0) * 100, 1) + '%');

    const annualEl = document.getElementById('annual-return');
    if (annualEl) {
        annualEl.textContent = formatPercent(state.annual_return_pct);
        annualEl.className = getColorClass(state.annual_return_pct);
    }

    const profitEl = document.getElementById('net-profit');
    if (profitEl) {
        profitEl.textContent = formatMoney(state.net_profit || 0);
        profitEl.className = getColorClass(state.net_profit);
    }
}

function updateEventLog(events) {
    const container = document.getElementById('event-log');
    if (!container) return;

    if (!events || events.length === 0) {
        container.innerHTML = '<div class="dim">No events yet...</div>';
        return;
    }

    // Take last 20 events (matching terminal display)
    const recentEvents = events.slice(-20).reverse();

    container.innerHTML = recentEvents.map(e => {
        let className = '';
        if (e.includes('REGIME') || e.includes('SHIFT')) className = 'event-regime';
        else if (e.includes('REBALANCE')) className = 'event-rebalance';
        else if (e.includes('CHECKPOINT')) className = 'event-checkpoint';
        else if (e.includes('ERROR')) className = 'event-error';

        return `<div class="${className}">${escapeHtml(e)}</div>`;
    }).join('');
}

// ═══════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function setTextWithClass(id, value, className) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = value;
        el.className = className;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateConnectionStatus(connected) {
    const indicator = document.getElementById('connection-indicator');
    const text = document.getElementById('connection-text');
    const lastUpdate = document.getElementById('last-update');

    if (connected) {
        isConnected = true;
        reconnectAttempts = 0;
        indicator.className = 'connected';
        text.textContent = 'Connected';
        lastUpdateTime = new Date();
        lastUpdate.textContent = 'Updated: ' + lastUpdateTime.toLocaleTimeString();
    } else {
        isConnected = false;
        indicator.className = 'disconnected';
        text.textContent = 'Disconnected';
    }
}

// ═══════════════════════════════════════════════════════════════
// POLLING LOOP
// ═══════════════════════════════════════════════════════════════

async function poll() {
    try {
        const response = await fetch('/api/state');

        if (response.ok) {
            const state = await response.json();
            updateDashboard(state);
        } else {
            console.warn('Poll response not OK:', response.status);
            updateConnectionStatus(false);
        }
    } catch (e) {
        console.error('Poll error:', e);
        updateConnectionStatus(false);

        reconnectAttempts++;
        if (reconnectAttempts > MAX_RECONNECT_ATTEMPTS) {
            console.log('Max reconnect attempts reached, slowing down polling');
            setTimeout(poll, RECONNECT_DELAY);
            return;
        }
    }

    setTimeout(poll, POLL_INTERVAL);
}

// ═══════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    console.log('MAFIA Dashboard initialized');
    updateConnectionStatus(false);
    poll();
});
