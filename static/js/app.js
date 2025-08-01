document.addEventListener('DOMContentLoaded', () => {
    // 전역 변수 및 상태 관리
    const state = {
        currentRunId: null,
        currentStockCode: null,
        currentStockName: null,
        charts: {}
    };

    // DOM 요소 캐싱
    const elements = {
        runListTableBody: document.querySelector('#run-list-table tbody'),
        performanceTableBody: document.querySelector('#performance-table tbody'),
        tradedStocksTableBody: document.querySelector('#traded-stocks-table tbody'),
        perfChartTitle: document.getElementById('performance-chart-title'),
        perfTableTitle: document.getElementById('performance-table-title'),
        tradedStocksTitle: document.getElementById('traded-stocks-title'),
        dailyChartTitle: document.getElementById('daily-chart-title'),
        minuteChartTitle: document.getElementById('minute-chart-title'),
        loadingOverlay: document.getElementById('loading-overlay')
    };

    // 로딩 오버레이 제어 함수
    const showLoading = () => elements.loadingOverlay.classList.remove('hidden');
    const hideLoading = () => elements.loadingOverlay.classList.add('hidden');

    // 숫자 및 날짜 포맷팅 헬퍼
    const formatPercent = (value) => value !== null ? `${Number(value).toFixed(2)}%` : 'N/A';
    const formatNumber = (value) => value !== null ? Number(value).toLocaleString() : 'N/A';
    const formatDate = (dateString) => dateString ? dateString.split('T')[0] : 'N/A';

    // 테이블 렌더링 함수
    function renderRunList(runs) {
        elements.runListTableBody.innerHTML = '';
        runs.forEach(run => {
            const row = document.createElement('tr');
            
            // ⬇️ 수정된 부분: run.runId -> run.run_id
            row.dataset.runId = run.run_id; 
            
            row.innerHTML = `
                <td>${run.run_id}</td> 
                <td>${formatDate(run.start_date)} ~ ${formatDate(run.end_date)}</td>
                <td>${run.strategy_daily || 'N/A'} /<br>${run.strategy_minute || 'N/A'}</td>
                <td class="align-right ${run.cumulative_return > 0 ? 'text-green' : 'text-red'}">${formatPercent(run.cumulative_return)}</td>
                <td class="align-right ${run.annualized_return > 0 ? 'text-green' : 'text-red'}">${formatPercent(run.annualized_return)}</td>
                <td class.align-right text-red">${formatPercent(run.max_drawdown)}</td>
            `;
            elements.runListTableBody.appendChild(row);
        });
    }
    
    // 차트 생성 및 업데이트 함수
    function createOrUpdateChart(chartId, config) {
        if (state.charts[chartId]) {
            state.charts[chartId].destroy();
        }
        const ctx = document.getElementById(chartId).getContext('2d');
        state.charts[chartId] = new Chart(ctx, config);
    }
    
    // --- 이벤트 핸들러 (안정성 강화를 위해 try...finally 추가) ---
    async function handleRunSelected(runId) {
        if (state.currentRunId === runId) return;
        
        showLoading();
        try {
            state.currentRunId = runId;

            document.querySelectorAll('#run-list-table tbody tr').forEach(tr => {
                tr.classList.toggle('selected', tr.dataset.runId === runId.toString());
            });

            elements.perfChartTitle.innerText = `누적 수익률 (Run ID: ${runId})`;
            elements.perfTableTitle.innerText = `일별 성능 (Run ID: ${runId})`;
            elements.tradedStocksTitle.innerText = `매매 종목 목록 (Run ID: ${runId})`;
            elements.dailyChartTitle.innerText = '종목 일봉 차트 (N/A)';
            elements.minuteChartTitle.innerText = '선택 일자 분봉 차트 (N/A)';

            const [perfData, stocksData] = await Promise.all([
                fetch(`/api/performance/${runId}`).then(res => res.json()),
                fetch(`/api/traded-stocks/${runId}`).then(res => res.json())
            ]);
            
            elements.performanceTableBody.innerHTML = '';
            perfData.forEach(p => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${formatDate(p.date)}</td>
                    <td class="align-right ${p.daily_return > 0 ? 'text-green' : 'text-red'}">${formatPercent(p.daily_return)}</td>
                    <td class="align-right ${p.cumulative_return > 0 ? 'text-green' : 'text-red'}">${formatPercent(p.cumulative_return)}</td>
                    <td class="align-right text-red">${formatPercent(p.drawdown)}</td>
                `;
                elements.performanceTableBody.appendChild(row);
            });

            createOrUpdateChart('performanceChart', {
                type: 'line',
                data: {
                    labels: perfData.map(p => formatDate(p.date)),
                    datasets: [{
                        label: '누적 수익률 (%)',
                        data: perfData.map(p => p.cumulative_return),
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        tension: 0.1
                    }]
                },
                options: { scales: { x: { type: 'time', time: { unit: 'day' } } } }
            });

            elements.tradedStocksTableBody.innerHTML = '';
            if (stocksData.length > 0) {
                stocksData.forEach(s => {
                    const row = document.createElement('tr');
                    row.dataset.stockCode = s.stock_code;
                    row.dataset.stockName = s.stock_name;
                    row.innerHTML = `
                        <td>${s.stock_name} (${s.stock_code})</td>
                        <td class="align-right">${s.trade_count}</td>
                        <td class="align-right ${s.total_realized_profit_loss > 0 ? 'text-green' : 'text-red'}">${formatNumber(s.total_realized_profit_loss)}</td>
                        <td class="align-right ${s.avg_return_per_trade > 0 ? 'text-green' : 'text-red'}">${formatPercent(s.avg_return_per_trade)}</td>
                    `;
                    elements.tradedStocksTableBody.appendChild(row);
                });
                const firstStock = stocksData[0];
                await handleStockSelected(firstStock.stock_code, firstStock.stock_name);
            } else {
                 if(state.charts['dailyChart']) state.charts['dailyChart'].destroy();
                 if(state.charts['minuteChart']) state.charts['minuteChart'].destroy();
            }
        } catch (error) {
            console.error('Error in handleRunSelected:', error);
            alert('백테스트 상세 정보를 불러오는 중 오류가 발생했습니다.');
        } finally {
            hideLoading();
        }
    }
    
    async function handleStockSelected(stockCode, stockName) {
        if (state.currentStockCode === stockCode) return;
        showLoading();
        try {
            state.currentStockCode = stockCode;
            state.currentStockName = stockName;
            
            document.querySelectorAll('#traded-stocks-table tbody tr').forEach(tr => {
                tr.classList.toggle('selected', tr.dataset.stockCode === stockCode);
            });
            
            elements.dailyChartTitle.innerText = `종목 일봉 차트 (${stockName} - ${stockCode})`;
            elements.minuteChartTitle.innerText = '선택 일자 분봉 차트 (N/A)';
    
            const chartData = await fetch(`/api/chart/daily/${state.currentRunId}/${stockCode}`).then(res => res.json());
    
            // --- ⬇️ 수정된 부분: d.date -> d.Date ---
            const ohlc = chartData.ohlcv.map(d => ({ x: new Date(d.Date).getTime(), o: d.open, h: d.high, l: d.low, c: d.close }));
            const volume = chartData.ohlcv.map(d => ({ x: new Date(d.Date).getTime(), y: d.volume, color: d.close >= d.open ? 'rgba(46, 204, 113, 0.5)' : 'rgba(231, 76, 60, 0.5)' }));
            
            // trade_datetime 키는 원래 소문자이므로 수정할 필요 없습니다.
            const buySignals = chartData.trades.filter(t => t.trade_type === 'BUY').map(t => ({ x: new Date(t.trade_datetime).getTime(), y: t.trade_price * 0.98 }));
            const sellSignals = chartData.trades.filter(t => t.trade_type === 'SELL').map(t => ({ x: new Date(t.trade_datetime).getTime(), y: t.trade_price * 1.02 }));
            
            createOrUpdateChart('dailyChart', {
                type: 'candlestick',
                data: { datasets: [{ label: 'OHLC', data: ohlc }, { label: '매수', type: 'scatter', data: buySignals, backgroundColor: 'green', pointStyle: 'triangle', radius: 7, rotation: 0 }, { label: '매도', type: 'scatter', data: sellSignals, backgroundColor: 'red', pointStyle: 'triangle', radius: 7, rotation: 180 }, { label: '거래량', type: 'bar', data: volume, yAxisID: 'yVolume' }] },
                options: { scales: { x: { type: 'time', time: { unit: 'day' } }, y: { position: 'left', title: { display: true, text: '가격' } }, yVolume: { position: 'right', title: { display: true, text: '거래량' }, grid: { drawOnChartArea: false }, ticks: { callback: value => `${value / 1000}k` } } } }
            });
    
            // Date 키는 대문자, trade_datetime 키는 소문자이므로 주의
            const dateToLoad = chartData.trades.length > 0 ? formatDate(chartData.trades[0].trade_datetime) : formatDate(chartData.ohlcv[0].Date);
            await handleDateSelected(dateToLoad);
        } catch (error) {
            console.error('Error in handleStockSelected:', error);
            alert('일봉 차트 데이터를 불러오는 중 오류가 발생했습니다.');
        } finally {
            hideLoading();
        }
    }
    
    async function handleDateSelected(tradeDate) {
        showLoading();
        try {
            elements.minuteChartTitle.innerText = `선택 일자 분봉 차트 (${state.currentStockName} - ${tradeDate})`;
            const url = `/api/chart/minute/${state.currentRunId}/${state.currentStockCode}?trade_date=${tradeDate}`;
            const chartData = await fetch(url).then(res => res.json());
    
            // 기존 차트가 있으면 삭제
            if (state.charts['minuteChartPrev']) state.charts['minuteChartPrev'].destroy();
            if (state.charts['minuteChartCurrent']) state.charts['minuteChartCurrent'].destroy();
    
            if (!chartData.ohlcv || chartData.ohlcv.length === 0) {
                elements.minuteChartTitle.innerText += ' (데이터 없음)';
                return;
            }
    
            // --- ⬇️ 데이터를 날짜별로 분리 ---
            const dataByDate = {};
            chartData.ohlcv.forEach(d => {
                const dateStr = new Date(d.Datetime).toISOString().split('T')[0];
                if (!dataByDate[dateStr]) {
                    dataByDate[dateStr] = [];
                }
                dataByDate[dateStr].push(d);
            });
    
            const dates = Object.keys(dataByDate).sort(); // 날짜순으로 정렬
    
            // --- ⬇️ 각 날짜에 대해 별도의 차트 생성 ---
            const createMinuteChart = (canvasId, dateKey) => {
                const dayData = dataByDate[dateKey];
                if (!dayData) return;
    
                const ohlc = dayData.map(d => ({ x: new Date(d.Datetime).getTime(), o: d.open, h: d.high, l: d.low, c: d.close }));
                const volume = dayData.map(d => ({ x: new Date(d.Datetime).getTime(), y: d.volume, color: d.close >= d.open ? 'rgba(239, 83, 80, 0.7)' : 'rgba(38, 166, 154, 0.7)' }));
                
                // 차트의 시작(09:00)과 종료(15:30) 시간 설정
                const dayStart = new Date(dateKey);
                dayStart.setHours(9, 0, 0, 0);
                const dayEnd = new Date(dateKey);
                dayEnd.setHours(15, 30, 0, 0);
    
                const config = {
                    type: 'candlestick',
                    data: { datasets: [{ label: 'OHLC', data: ohlc }, { label: '거래량', type: 'bar', data: volume, yAxisID: 'yVolume' }] },
                    options: {
                        plugins: {
                            title: { display: true, text: dateKey }, // 차트 위에 날짜 표시
                            legend: { display: false }
                        },
                        scales: {
                            x: {
                                type: 'time',
                                time: { unit: 'minute', tooltipFormat: 'HH:mm' },
                                min: dayStart.getTime(),
                                max: dayEnd.getTime(),
                            },
                            y: { position: 'left' },
                            yVolume: { position: 'right', grid: { drawOnChartArea: false }, ticks: { callback: v => `${Math.round(v/1000)}k` } }
                        }
                    }
                };
                createOrUpdateChart(canvasId, config);
            };
            
            // 전일과 당일 차트 생성
            if (dates.length > 0) createMinuteChart('minuteChartPrev', dates[0]);
            if (dates.length > 1) createMinuteChart('minuteChartCurrent', dates[1]);
            
        } catch (error) {
            console.error('Error in handleDateSelected:', error);
            alert('분봉 차트 데이터를 불러오는 중 오류가 발생했습니다.');
        } finally {
            hideLoading();
        }
    }

    // 이벤트 리스너 등록
    elements.runListTableBody.addEventListener('click', (e) => {
        const row = e.target.closest('tr');
        if (row && row.dataset.runId) {
            handleRunSelected(parseInt(row.dataset.runId, 10));
        }
    });

    elements.tradedStocksTableBody.addEventListener('click', (e) => {
        const row = e.target.closest('tr');
        if (row && row.dataset.stockCode) {
            handleStockSelected(row.dataset.stockCode, row.dataset.stockName);
        }
    });

    // 초기 데이터 로드
    async function initialize() {
        showLoading();
        try {
            const runs = await fetch('/api/runs').then(res => res.json());
            renderRunList(runs);
            
            if (runs.length > 0) {
                await handleRunSelected(runs[0].run_id);
            }
        } catch (error) {
            console.error('Error during initialization:', error);
            alert('초기 데이터를 불러오는 중 오류가 발생했습니다.');
        } finally {
            hideLoading();
        }
    }

    initialize();
});