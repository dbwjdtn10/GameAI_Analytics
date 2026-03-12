-- GameAI Analytics - 분석 쿼리 모음

-- ============================================================
-- 1. 기본 통계
-- ============================================================

-- 전체 이탈률
SELECT
    COUNT(*) AS total_users,
    SUM(CASE WHEN is_churned THEN 1 ELSE 0 END) AS churned_users,
    ROUND(AVG(is_churned::int) * 100, 2) AS churn_rate_pct
FROM players;

-- 장르별 이탈률
SELECT
    game_genre,
    COUNT(*) AS user_count,
    ROUND(AVG(is_churned::int) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(play_time_hours), 1) AS avg_playtime,
    ROUND(AVG(player_level), 1) AS avg_level
FROM players
GROUP BY game_genre
ORDER BY churn_rate_pct DESC;

-- 난이도별 이탈률
SELECT
    game_difficulty,
    COUNT(*) AS user_count,
    ROUND(AVG(is_churned::int) * 100, 2) AS churn_rate_pct
FROM players
GROUP BY game_difficulty
ORDER BY churn_rate_pct DESC;

-- ============================================================
-- 2. 유저 세그먼트 분석
-- ============================================================

-- 플레이타임 기반 세그먼트
SELECT
    CASE
        WHEN play_time_hours < 5 THEN '초보 (< 5h)'
        WHEN play_time_hours < 20 THEN '일반 (5-20h)'
        WHEN play_time_hours < 50 THEN '활발 (20-50h)'
        ELSE '하드코어 (50h+)'
    END AS playtime_segment,
    COUNT(*) AS user_count,
    ROUND(AVG(is_churned::int) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(in_game_purchases), 2) AS avg_purchases,
    ROUND(AVG(sessions_per_week), 1) AS avg_sessions
FROM players
GROUP BY playtime_segment
ORDER BY churn_rate_pct DESC;

-- 레벨 구간별 이탈률 (10레벨 단위)
SELECT
    (player_level / 10) * 10 AS level_range_start,
    COUNT(*) AS user_count,
    ROUND(AVG(is_churned::int) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(play_time_hours), 1) AS avg_playtime
FROM players
GROUP BY level_range_start
ORDER BY level_range_start;

-- ============================================================
-- 3. 고위험 유저 조회
-- ============================================================

-- 최근 예측에서 고위험 유저 목록
SELECT
    p.player_id,
    p.game_genre,
    p.player_level,
    p.play_time_hours,
    pr.churn_probability,
    pr.risk_level,
    pr.segment
FROM players p
JOIN predictions pr ON p.player_id = pr.player_id
WHERE pr.risk_level IN ('high', 'critical')
ORDER BY pr.churn_probability DESC
LIMIT 100;

-- 위험 등급별 유저 분포
SELECT
    pr.risk_level,
    COUNT(*) AS user_count,
    ROUND(AVG(pr.churn_probability) * 100, 2) AS avg_churn_prob_pct,
    ROUND(AVG(p.play_time_hours), 1) AS avg_playtime,
    ROUND(AVG(p.in_game_purchases), 2) AS avg_purchases
FROM predictions pr
JOIN players p ON pr.player_id = p.player_id
WHERE pr.predicted_at = (
    SELECT MAX(predicted_at) FROM predictions
)
GROUP BY pr.risk_level
ORDER BY avg_churn_prob_pct DESC;

-- ============================================================
-- 4. 구매 분석
-- ============================================================

-- 구매 유저 vs 비구매 유저 이탈률 비교
SELECT
    CASE WHEN in_game_purchases > 0 THEN '구매 유저' ELSE '비구매 유저' END AS purchase_group,
    COUNT(*) AS user_count,
    ROUND(AVG(is_churned::int) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(play_time_hours), 1) AS avg_playtime,
    ROUND(AVG(player_level), 1) AS avg_level
FROM players
GROUP BY purchase_group;

-- 구매 횟수 구간별 이탈률
SELECT
    CASE
        WHEN in_game_purchases = 0 THEN '0회'
        WHEN in_game_purchases BETWEEN 1 AND 5 THEN '1-5회'
        WHEN in_game_purchases BETWEEN 6 AND 15 THEN '6-15회'
        ELSE '16회 이상'
    END AS purchase_range,
    COUNT(*) AS user_count,
    ROUND(AVG(is_churned::int) * 100, 2) AS churn_rate_pct
FROM players
GROUP BY purchase_range
ORDER BY churn_rate_pct DESC;

-- ============================================================
-- 5. 모델 성능 추이
-- ============================================================

-- 모델 학습 이력
SELECT
    model_name,
    model_type,
    ROUND(auc_roc::numeric, 4) AS auc_roc,
    ROUND(f1_score::numeric, 4) AS f1_score,
    ROUND(accuracy::numeric, 4) AS accuracy,
    is_active,
    trained_at
FROM model_registry
ORDER BY trained_at DESC
LIMIT 10;

-- 현재 활성 모델
SELECT * FROM model_registry WHERE is_active = TRUE;
