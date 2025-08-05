import sys
import os
import pandas as pd
import numpy as np # <-- numpy 추가
from datetime import date
from typing import List, Dict, Any, Optional # <--- Optional 추가

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- 경로 설정 ---
# 1. 현재 파일(webapp/main.py)이 있는 디렉토리의 절대 경로를 구합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 프로젝트 루트 경로를 계산하고 sys.path에 추가합니다.
project_root = os.path.abspath(os.path.join(BASE_DIR, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 기존 비즈니스 로직 매니저 임포트
from manager.app_manager import AppManager

# --- FastAPI 앱 설정 ---
app = FastAPI()
# 3. ⬇️ BASE_DIR를 사용하여 static과 templates 폴더의 절대 경로를 지정합니다.
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# AppManager 인스턴스를 전역으로 생성하여 재사용
try:
    app_manager = AppManager()
    print("✅ AppManager가 성공적으로 초기화되었습니다.")
except Exception as e:
    print(f"❌ AppManager 초기화 실패: {e}")
    app_manager = None

# --- Pydantic 응답 모델 정의 (API 문서화를 위함) ---
class BacktestRun(BaseModel):
    run_id: int  # <--- 이 라인을 맨 위에 추가하세요.
    start_date: Optional[date] = None # date도 Optional로 변경하는 것이 안전
    end_date: Optional[date] = None
    # ⬇️ 모든 숫자 필드를 Optional로 변경
    initial_capital: Optional[float] = None
    final_capital: Optional[float] = None
    total_profit_loss: Optional[float] = None
    cumulative_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    annualized_return: Optional[float] = None
    strategy_daily: Optional[str] = None
    strategy_minute: Optional[str] = None

# --- API 엔드포인트 ---
class PerformanceData(BaseModel):
    performance_id: Optional[int] = None
    run_id: Optional[int] = None
    date: date 
    end_capital: Optional[float] = None
    daily_return: Optional[float] = None
    daily_profit_loss: Optional[float] = None
    cumulative_return: Optional[float] = None
    drawdown: Optional[float] = None

class TradedStockSummary(BaseModel):
    stock_code: Optional[str] = None
    stock_name: Optional[str] = None
    trade_count: Optional[int] = None
    total_realized_profit_loss: Optional[float] = None
    avg_return_per_trade: Optional[float] = None

# =================================================================
# 💻 페이지 렌더링 라우트
# =================================================================
@app.get("/", response_class=HTMLResponse)
async def route_root(request: Request):
    # 이제 루트 경로는 화려한 홈페이지(index.html)를 보여줍니다.
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def route_dashboard(request: Request):
    return templates.TemplateResponse("dashboard/index.html", {"request": request})

# --- 인증 ---
@app.get("/auth/login", response_class=HTMLResponse)
async def route_login(request: Request):
    return templates.TemplateResponse("auth/login.html", {"request": request})

@app.get("/auth/signup", response_class=HTMLResponse)
async def route_signup(request: Request):
    return templates.TemplateResponse("auth/signup.html", {"request": request})

@app.get("/auth/forgot-password", response_class=HTMLResponse)
async def route_forgot_password(request: Request):
    return templates.TemplateResponse("auth/forgot_password.html", {"request": request})

# --- 매매 현황 ---
@app.get("/trading/monitoring", response_class=HTMLResponse)
async def route_trading_monitoring(request: Request):
    return templates.TemplateResponse("trading/monitoring.html", {"request": request})

@app.get("/trading/analysis", response_class=HTMLResponse)
async def route_trading_analysis(request: Request):
    return templates.TemplateResponse("trading/analysis.html", {"request": request})

# --- 백테스팅 ---
@app.get("/backtesting/run", response_class=HTMLResponse)
async def route_backtest_run(request: Request):
    return templates.TemplateResponse("backtesting/run.html", {"request": request})

@app.get("/backtesting/results", response_class=HTMLResponse)
async def route_backtest_results(request: Request):
    return templates.TemplateResponse("backtesting/results.html", {"request": request})

# --- 설정 ---
@app.get("/settings/universe", response_class=HTMLResponse)
async def route_settings_universe(request: Request):
    return templates.TemplateResponse("settings/universe.html", {"request": request})

@app.get("/settings/notifications", response_class=HTMLResponse)
async def route_settings_notifications(request: Request):
    return templates.TemplateResponse("settings/notifications.html", {"request": request})

@app.get("/settings/edit-profile", response_class=HTMLResponse)
async def route_settings_edit_profile(request: Request):
    return templates.TemplateResponse("settings/edit_profile.html", {"request": request})
# --- 결제 ---
@app.get("/billing/usage", response_class=HTMLResponse)
async def route_billing_usage(request: Request):
    return templates.TemplateResponse("billing/usage.html", {"request": request})

@app.get("/billing/methods", response_class=HTMLResponse)
async def route_billing_methods(request: Request):
    return templates.TemplateResponse("billing/methods.html", {"request": request})

@app.get("/billing/history", response_class=HTMLResponse)
async def route_billing_history(request: Request):
    return templates.TemplateResponse("billing/history.html", {"request": request})

# --- 고객지원 ---
@app.get("/support/announcements", response_class=HTMLResponse)
async def route_support_announcements(request: Request):
    return templates.TemplateResponse("support/announcements.html", {"request": request})

@app.get("/support/faq", response_class=HTMLResponse)
async def route_support_faq(request: Request):
    return templates.TemplateResponse("support/faq.html", {"request": request})

@app.get("/support/inquiry", response_class=HTMLResponse)
async def route_support_inquiry(request: Request):
    return templates.TemplateResponse("support/inquiry.html", {"request": request})

# --- 관리자 ---
@app.get("/admin/user-management", response_class=HTMLResponse)
async def route_admin_user_management(request: Request):
    return templates.TemplateResponse("admin/user_management.html", {"request": request})

# 
@app.get("/admin/billing-management", response_class=HTMLResponse)
async def route_admin_billing_management(request: Request):
    return templates.TemplateResponse("admin/billing_management.html", {"request": request})

@app.get("/admin/system-monitoring", response_class=HTMLResponse)
async def route_admin_system_monitoring(request: Request):
    return templates.TemplateResponse("admin/system_monitoring.html", {"request": request})
# =================================================================
# 📈 API 데이터 라우트 (기존 코드 유지)
# =================================================================

@app.get("/api/runs", response_model=List[BacktestRun])
async def get_backtest_runs():
    """모든 백테스트 실행 목록을 조회합니다."""
    if not app_manager:
        return JSONResponse(status_code=500, content={"message": "AppManager가 초기화되지 않았습니다."})

    runs_df = app_manager.get_backtest_runs()
    if runs_df.empty:
        return []
    
    # numpy의 무한대(inf, -inf)와 NaN 값을 None으로 변환합니다.
    runs_df = runs_df.replace([np.inf, -np.inf], None)

    # Pandas의 to_dict는 NaN을 None으로 자동 변환해주지만,
    # 명시적으로 모든 빈 값을 None으로 한번 더 처리하여 안정성을 높입니다.
    runs_df = runs_df.where(pd.notna(runs_df), None)

    return runs_df.to_dict('records')

@app.get("/api/performance/{run_id}", response_model=List[PerformanceData]) # <--- response_model 추가
async def get_performance(run_id: int):
    """특정 백테스트의 일별 성과 데이터를 조회합니다."""
    performance_df = app_manager.get_backtest_performance(run_id)
    performance_df = performance_df.replace([np.inf, -np.inf], None).where(pd.notna(performance_df), None)

    # ⬇️ return 방식을 JSONResponse 없이 바로 반환하도록 변경
    return performance_df.to_dict('records') 

@app.get("/api/traded-stocks/{run_id}", response_model=List[TradedStockSummary]) # <--- response_model 추가
async def get_traded_stocks_summary(run_id: int):
    """특정 백테스트의 매매 종목 요약 정보를 조회합니다."""
    summary_df = app_manager.get_traded_stocks_summary(run_id)
    summary_df = summary_df.replace([np.inf, -np.inf], None).where(pd.notna(summary_df), None)

    # ⬇️ return 방식을 JSONResponse 없이 바로 반환하도록 변경
    return summary_df.to_dict('records')

@app.get("/api/chart/daily/{run_id}/{stock_code}")
async def get_daily_chart_data(run_id: int, stock_code: str):
    """일봉 차트 데이터를 조회합니다 (OHLCV + 매매 내역)."""
    # 임시로 AppManager를 사용하여 데이터 로드 (실제로는 Model 계층을 통하는 것이 좋음)
    run_info = app_manager.db_manager.fetch_backtest_run(run_id=run_id)
    if run_info.empty:
        return JSONResponse(status_code=404, content={"message": "Run ID not found"})
        
    start_date = pd.to_datetime(run_info['start_date'].iloc[0]).date()
    end_date = pd.to_datetime(run_info['end_date'].iloc[0]).date()
    
    params_daily = pd.read_json(run_info['params_json_daily'].iloc[0], typ='series').to_dict() if run_info['params_json_daily'].iloc[0] else {}
    params_daily['strategy_name'] = run_info['strategy_daily'].iloc[0]

    ohlcv_df = app_manager.get_daily_ohlcv_with_indicators(stock_code, start_date, end_date, params_daily)
    trades_df = app_manager.get_backtest_trades(run_id)
    trades_df = trades_df[trades_df['stock_code'] == stock_code]

    # --- 데이터 정제 코드 추가 ---
    ohlcv_df = ohlcv_df.replace([np.inf, -np.inf], None).where(pd.notna(ohlcv_df), None)
    trades_df = trades_df.replace([np.inf, -np.inf], None).where(pd.notna(trades_df), None)
    
    ohlcv_df.reset_index(inplace=True)
    
    return {
        "ohlcv": ohlcv_df.to_dict('records'),
        "trades": trades_df.to_dict('records')
    }

@app.get("/api/chart/minute/{run_id}/{stock_code}")
async def get_minute_chart_data(run_id: int, stock_code: str, trade_date: date):
    """분봉 차트 데이터를 조회합니다 (OHLCV + 매매 내역)."""
    run_info = app_manager.db_manager.fetch_backtest_run(run_id=run_id)
    if run_info.empty:
        return JSONResponse(status_code=404, content={"message": "Run ID not found"})
    
    params_minute = pd.read_json(run_info['params_json_minute'].iloc[0], typ='series').to_dict() if run_info['params_json_minute'].iloc[0] else {}
    params_minute['strategy_name'] = run_info['strategy_minute'].iloc[0]

    ohlcv_df = app_manager.get_minute_ohlcv_with_indicators(stock_code, trade_date, params_minute)
    
    trades_df = app_manager.get_backtest_trades(run_id)
    trades_for_date = trades_df[
        (trades_df['stock_code'] == stock_code) &
        (pd.to_datetime(trades_df['trade_datetime']).dt.date == trade_date)
    ]
    # --- 데이터 정제 코드 추가 ---
    ohlcv_df = ohlcv_df.replace([np.inf, -np.inf], None).where(pd.notna(ohlcv_df), None)
    trades_for_date = trades_for_date.replace([np.inf, -np.inf], None).where(pd.notna(trades_for_date), None)

    ohlcv_df.reset_index(inplace=True)
    
    return {
        "ohlcv": ohlcv_df.to_dict('records'),
        "trades": trades_for_date.to_dict('records')
    }

# --- 서버 실행 (개발용) ---
if __name__ == "__main__":
    import uvicorn
    # 외부 접근을 허용하려면 host="0.0.0.0"으로 설정
    uvicorn.run(app, host="127.0.0.1", port=8000)