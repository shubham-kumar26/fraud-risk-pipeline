
CREATE TABLE transactions (
     idx int,
    trans_date_trans_time TIMESTAMP,
    cc_num BIGINT,
    merchant VARCHAR(100),
    category VARCHAR(50),
    amt NUMERIC(10,2),
    first VARCHAR(50),
    last VARCHAR(50),
    gender CHAR(1),
    street VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(5),
    zip VARCHAR(10),
    lat NUMERIC(9,6),
    long NUMERIC(9,6),
    city_pop INT,
    job VARCHAR(100),
    dob DATE,
    trans_num VARCHAR(50),
    unix_time BIGINT,
    merch_lat NUMERIC(9,6),
    merch_long NUMERIC(9,6),
    is_fraud BOOLEAN);

select * from transactions;
SELECT column_name FROM information_schema.columns WHERE table_name = 'transactions';

\copy transactions(idx, trans_date_trans_time, cc_num, merchant, category, amt,
first, last, gender, street, city, state, zip, lat, long, city_pop,
job, dob, trans_num, unix_time, merch_lat, merch_long,
is_fraud) FROM 'C:/Users/HP/Downloads/CREDIT/fraudTrain.csv/fraudTrain.csv' 
WITH (FORMAT csv, DELIMITER ',', HEADER true, QUOTE '"')

SELECT COUNT(*) FROM transactions;

SELECT COUNT(*) FROM transactions WHERE is_fraud = true;



-- fraud rate

SELECT category, 
       COUNT(*) AS total,
       SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count,
       ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY category
ORDER BY fraud_rate_pct DESC;

---fraud by age group

SELECT 
    CASE 
        WHEN AGE(CURRENT_DATE, dob) < INTERVAL '25 years' THEN 'Under 25'
        WHEN AGE(CURRENT_DATE, dob) < INTERVAL '40 years' THEN '25-39'
        WHEN AGE(CURRENT_DATE, dob) < INTERVAL '60 years' THEN '40-59'
        ELSE '60+'
    END AS age_group,
    COUNT(*) AS total,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY age_group
ORDER BY fraud_rate_pct DESC;

-- distance  between customer and merchant

SELECT 
    is_fraud,
    ROUND(AVG(ABS(lat - merch_lat) + ABS(long - merch_long))::numeric, 4) AS avg_location_diff
FROM transactions
GROUP BY is_fraud;



SELECT 
    is_fraud,
    ROUND(AVG(
        3959 * acos(
            cos(radians(lat)) * cos(radians(merch_lat)) * 
            cos(radians(merch_long) - radians(long)) + 
            sin(radians(lat)) * sin(radians(merch_lat))
        )
    )::numeric, 2) AS avg_distance_miles
FROM transactions
GROUP BY is_fraud;



--transaction count by hour:

SELECT 
    EXTRACT(HOUR FROM trans_date_trans_time) AS hour_of_day,
    COUNT(*) AS total
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;






--fraud count


SELECT 
    EXTRACT(HOUR FROM trans_date_trans_time) AS hour_of_day,
    COUNT(*) AS total,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;


--- hourly fraud count with percentage

SELECT 
    EXTRACT(HOUR FROM trans_date_trans_time) AS hour_of_day,
    COUNT(*) AS total,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY fraud_rate_pct DESC;


--top 10 credit card with highet transcations

select cc_num,count(*) AS total_transactions
FROM transactions
GROUP BY cc_num
ORDER BY total_transactions DESC
LIMIT 10;

-- now ranking the top 10 credit cards and we can also see their fraud counts

SELECT 
    cc_num, 
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count,
    RANK() OVER (ORDER BY COUNT(*) DESC) AS rank_by_activity
FROM transactions
GROUP BY cc_num
ORDER BY rank_by_activity
LIMIT 10;






SELECT 
    cc_num,
    trans_date_trans_time,
    amt,
    LAG(amt) OVER (PARTITION BY cc_num ORDER BY trans_date_trans_time) AS prev_amt
FROM transactions
WHERE cc_num = 571365235126
ORDER BY trans_date_trans_time
LIMIT 20;


--flag actual spending spikes across ALL customers



WITH spending_pattern AS (
    SELECT 
        cc_num,
        trans_date_trans_time,
        amt,
        is_fraud,
        LAG(amt) OVER (PARTITION BY cc_num ORDER BY trans_date_trans_time) AS prev_amt
    FROM transactions
)
SELECT *
FROM spending_pattern
WHERE amt > prev_amt * 5
ORDER BY amt DESC
LIMIT 20;



SELECT 
    is_fraud,
    ROUND(AVG(amt)::numeric, 2) AS avg_amount,
    ROUND(MIN(amt)::numeric, 2) AS min_amount,
    ROUND(MAX(amt)::numeric, 2) AS max_amount
FROM transactions
GROUP BY is_fraud;






