UPDATE daily_price dp
LEFT JOIN (
    -- 각 종목의 이전 거래일 종가를 가져오기 위한 서브쿼리
    SELECT
        stock_code,
        date,
        close AS prev_close
    FROM
        daily_price
) prev_dp ON dp.stock_code = prev_dp.stock_code
           AND prev_dp.date = (SELECT MAX(date) FROM daily_price WHERE stock_code = dp.stock_code AND date < dp.date)
SET
    dp.trading_value = dp.close * dp.volume, -- 거래대금 계산
    dp.change_rate = CASE
                        WHEN prev_dp.prev_close IS NOT NULL AND prev_dp.prev_close <> 0
                        THEN ((dp.close - prev_dp.prev_close) / prev_dp.prev_close) * 100 -- 등락률 계산
                        ELSE 0.0000 -- 이전일 데이터가 없거나 0인 경우 0으로 설정
                     END;