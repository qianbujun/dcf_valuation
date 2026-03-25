import os
import json
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import tushare as ts

"""
dcf+反向dcf网页版本
"""

# ==============================================================================
#  初始化配置 (OpenAI & Tushare)
# ==============================================================================
app = Flask(__name__)

# --- AI 配置 ---@
API_KEY = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print(f"AI客户端初始化失败: {e}")
    client = None

# --- Tushare 配置 ---
TS_TOKEN = "XXXXXXXXXXXXXXXXXXXXXXXXXX"
try:
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
except Exception as e:
    print(f"Tushare 初始化失败: {e}")
    pro = None


# ==============================================================================
#  数据自动化抓取引擎 (Tushare + 同花顺 F10)
# ==============================================================================
def check_and_update_data():
    """管理并更新本地的 A股股票基础信息表 (stock_basic.csv)"""
    data_file = "stock_basic.csv"
    need_update = False

    if not os.path.exists(data_file):
        need_update = True
    else:
        file_mtime = os.path.getmtime(data_file)
        file_date = datetime.fromtimestamp(file_mtime).date()
        if file_date < date.today():
            need_update = True

    if need_update and pro:
        try:
            df = pro.stock_basic(exchange='', list_status='L',
                                 fields='ts_code,symbol,name,area,industry,list_date,market')
            if not df.empty:
                df.to_csv(data_file, index=False, encoding='utf-8-sig')
                return df
        except Exception as e:
            print(f"更新 stock_basic 失败: {e}")

    if os.path.exists(data_file):
        return pd.read_csv(data_file, dtype={'symbol': str, 'ts_code': str})
    return None


def resolve_stock_query(query: str):
    """根据输入的 6位代码 或 公司名称 解析出 ts_code 和 标准全称"""
    df = check_and_update_data()
    if df is None or df.empty:
        return None, None

    if query.isdigit() and len(query) == 6:
        res = df[df['symbol'] == query]
    else:
        res = df[df['name'] == query]

    if not res.empty:
        return res.iloc[0]['ts_code'], res.iloc[0]['name']
    return None, None


def get_company_full_name_by_selector(stock_code_6digit: str):
    """从同花顺爬取公司全名及简介"""
    url = f"https://basic.10jqka.com.cn/{stock_code_6digit}/company.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://basic.10jqka.com.cn/'
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        element = soup.select_one('.video-btn-box-tr td:nth-of-type(2) span')
        if element:
            return element.get_text(strip=True)
    except Exception as e:
        print(f"爬取F10公司简介失败: {e}")
    return ""


def fetch_last_x_years_annual_reports(ts_code: str, x: int = 3, target_report_type: str = '1'):
    """
    从 Tushare 获取三大表最近 X 年(1231)的数据并格式化为文本
    :param ts_code: 股票代码，如 '000001.SZ'
    :param x: 获取最近 X 年的年报，默认为 3 年
    :param target_report_type: 报表类型。'1'为合并报表(估值推荐使用)，'6'或对应代码为母公司报表
    """
    if not pro: return ""

    start_dt = (datetime.now() - timedelta(days=365 * (x + 2))).strftime('%Y%m%d')
    end_dt = datetime.now().strftime('%Y%m%d')

    def get_clean_statement(api_func, **kwargs):
        df = api_func(**kwargs)
        if df.empty: return pd.DataFrame()

        # 1. 筛选年报(12月31日)
        df = df[df['end_date'].str.endswith('1231')]

        # 2. 筛选报表类型 (重点在这里)
        if 'report_type' in df.columns:
            df = df[df['report_type'] == target_report_type]

        # 3. 按报告期和公告日期降序排列并去重（保留最新更正版本）
        if not df.empty and 'ann_date' in df.columns:
            df = df.sort_values(['end_date', 'ann_date'], ascending=[False, False])
            df = df.drop_duplicates(subset=['end_date'], keep='first')
        else:
            df = df.sort_values('end_date', ascending=False)
            df = df.drop_duplicates(subset=['end_date'], keep='first')

        return df.head(x)

    try:
        # 依次获取三大表数据
        df_inc = get_clean_statement(pro.income, ts_code=ts_code, start_date=start_dt, end_date=end_dt)
        df_bs = get_clean_statement(pro.balancesheet, ts_code=ts_code, start_date=start_dt, end_date=end_dt)
        df_cf = get_clean_statement(pro.cashflow, ts_code=ts_code, start_date=start_dt, end_date=end_dt)

        all_end_dates = set()
        for df in [df_inc, df_bs, df_cf]:
            if not df.empty:
                all_end_dates.update(df['end_date'].tolist())

        target_dates = sorted(list(all_end_dates), reverse=True)[:x]
        report_texts = []

        report_name = "合并报表" if target_report_type == '1' else "母公司/单体报表"

        for edate in target_dates:
            year_text = f"=========== 【{edate[:4]}年 年度报告 ({report_name} 报告期:{edate})】 ==========="

            if not df_inc.empty and edate in df_inc['end_date'].values:
                inc_dict = df_inc[df_inc['end_date'] == edate].iloc[0].drop(
                    labels=['ts_code', 'end_date', 'report_type'], errors='ignore'
                ).dropna().to_dict()
                year_text += f"\n[利润表]\n{json.dumps(inc_dict, ensure_ascii=False)}"

            if not df_bs.empty and edate in df_bs['end_date'].values:
                bs_dict = df_bs[df_bs['end_date'] == edate].iloc[0].drop(
                    labels=['ts_code', 'end_date', 'report_type', 'ann_date', 'f_ann_date'], errors='ignore'
                ).dropna().to_dict()
                year_text += f"\n\n[资产负债表]\n{json.dumps(bs_dict, ensure_ascii=False)}"

            if not df_cf.empty and edate in df_cf['end_date'].values:
                cf_dict = df_cf[df_cf['end_date'] == edate].iloc[0].drop(
                    labels=['ts_code', 'end_date', 'report_type', 'ann_date', 'f_ann_date'], errors='ignore'
                ).dropna().to_dict()
                year_text += f"\n\n[现金流量表]\n{json.dumps(cf_dict, ensure_ascii=False)}"

            report_texts.append(year_text)

        return "\n\n".join(report_texts)

    except Exception as e:
        print(f"获取财报发生错误: {e}")
        return ""


# ==============================================================================
#  核心财务计算函数
# ==============================================================================
def calculate_dcf_valuation(current_revenue, projection_years, revenue_growth_rates, ebit_margin_rates, tax_rate,
                            reinvestment_rate, discount_rate, terminal_growth_rate) -> dict:
    """正向 DCF 核心引擎"""
    projected_data = []
    last_revenue = current_revenue

    def get_rate(rate_list, index):
        if not rate_list: return 0.0
        if index < len(rate_list): return float(rate_list[index])
        return float(rate_list[-1])

    for i in range(projection_years):
        year_data = {'Year': i + 1}
        g_rate = get_rate(revenue_growth_rates, i)
        current_year_revenue = last_revenue * (1 + g_rate)
        year_data['Revenue'] = current_year_revenue

        margin = get_rate(ebit_margin_rates, i)
        ebit = current_year_revenue * margin
        year_data['EBIT'] = ebit

        nopat = ebit * (1 - tax_rate)
        year_data['NOPAT'] = nopat

        r_rate = get_rate(reinvestment_rate, i)
        reinvestment = nopat * r_rate
        fcff = nopat - reinvestment

        year_data['Reinvestment Rate'] = r_rate
        year_data['Reinvestment'] = reinvestment
        year_data['FCFF'] = fcff

        projected_data.append(year_data)
        last_revenue = current_year_revenue

    pv_of_fcff_sum = sum(data['FCFF'] / ((1 + discount_rate) ** (data['Year'] - 0.5)) for data in projected_data)
    for data in projected_data:
        data['PV of FCFF'] = data['FCFF'] / ((1 + discount_rate) ** (data['Year'] - 0.5))

    final_year_fcff = projected_data[-1]['FCFF']
    terminal_value = (final_year_fcff * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    pv_of_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)

    return {"enterprise_value": pv_of_fcff_sum + pv_of_terminal_value, "details": pd.DataFrame(projected_data)}


def calculate_reverse_dcf(current_market_cap, current_revenue, projection_years, ebit_margin_rates, tax_rate,
                          reinvestment_rate, discount_rate, terminal_growth_rate, total_debt, cash) -> dict:
    """反向 DCF 核心引擎 (二分查找)"""

    def get_market_cap_at_growth(g):
        res = calculate_dcf_valuation(current_revenue, projection_years, [g] * projection_years, ebit_margin_rates,
                                      tax_rate, reinvestment_rate, discount_rate, terminal_growth_rate)
        return (res['enterprise_value'] - total_debt + cash), res

    low, high, tolerance = -0.99, 5.00, 0.0001
    cap_at_low, res_low = get_market_cap_at_growth(low)
    cap_at_high, res_high = get_market_cap_at_growth(high)

    if current_market_cap <= cap_at_low: return {"implied_g": low, "details": res_low['details'],
                                                 "msg": "市值极低，触及下限"}
    if current_market_cap >= cap_at_high: return {"implied_g": high, "details": res_high['details'],
                                                  "msg": "市值极高，触及上限"}

    best_res, iterations = None, 0
    while high - low > tolerance:
        mid = (low + high) / 2
        calc_cap, res = get_market_cap_at_growth(mid)
        if calc_cap < current_market_cap:
            low = mid
        else:
            high = mid
        iterations += 1
        best_res = res

    return {"implied_g": (low + high) / 2, "details": best_res['details'], "msg": f"计算成功 (迭代 {iterations} 次)"}


# ==============================================================================
#  后端 API 路由
# ==============================================================================
@app.route('/api/autofetch', methods=['POST'])
def api_autofetch():
    """自动化获取 Tushare 数据接口"""
    data = request.json
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"status": "error", "message": "请输入股票简称或6位代码"})

    try:
        ts_code, stock_name = resolve_stock_query(query)
        if not ts_code:
            return jsonify({"status": "error", "message": f"未找到匹配的股票: {query}"})

        pure_code = ts_code.split('.')[0]
        comp_desc = get_company_full_name_by_selector(pure_code)
        if not comp_desc:
            comp_desc = f"{stock_name} (自动提取同花顺F10简介失败，您可以手动输入补充)"

        fin_text = fetch_last_x_years_annual_reports(ts_code, 2)
        if not fin_text:
            return jsonify({"status": "error",
                            "message": f"成功找到 {stock_name}({ts_code})，但未能获取到最近4年的完整年报数据，请确认其是否有公开数据。"})

        return jsonify({
            "status": "success",
            "company_name": stock_name,
            "company_desc": comp_desc,
            "financials": fin_text
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"数据获取异常: {str(e)}"})


@app.route('/api/generate', methods=['POST'])
def ai_generate():
    data = request.json
    try:
        proj_years = int(data.get('projection_years', 5))
    except (ValueError, TypeError):
        proj_years = 5

    # 动态生成符合要求长度的示例数组（如果传进来是10，数组长度就严格为10）
    example_growth = [round(max(0.15 - i * 0.01, 0.05), 2) for i in range(proj_years)]
    example_margin = [round(min(0.10 + i * 0.01, 0.25), 2) for i in range(proj_years)]
    example_reinv = [0.15] * proj_years

    prompt = f"""
你是一位顶级的金融分析师，任务是为一家公司进行DCF估值前的参数准备。
请根据提供的原始信息，深入分析该公司的行业特性和商业模式，并严格按照指定的JSON格式生成一个参数字典。

**绝对单位要求：**
所有的绝对财务数值请必须直接提取或换算为以**“万元 (10,000 RMB)”**为单位的纯数字输出！由于你接收到的是以元为单位的原始API数据字典，请你务必将其除以 10000 转换后再填入！

**核心预测逻辑与异常校验要求：**
1. **基期数据提取**：分析最新财报。如果发现财报数据严重缺失，请**务必将 `revenue` 设为 0（绝对不要自行编造微小数字）**，并在 `_comment` 中严厉提示！
2. **适用性警告（重点）**：如果判定该公司**目前处于亏损状态**，或者属于**强周期/重资产行业（如大宗商品、航运、房地产开发等）**，请务必在 `_comment` 字段的开头加上：【警告：该公司处于亏损状态或属于强周期行业，传统DCF估值模型可能完全不适用！】
3. **商业模式与资产轻重判定**：判断其属于轻资产、重资产还是科技/成长驱动型。
4. **再投资率 (Reinvestment Rate)**：轻资产极小(10%~30%)，重资产适中或极高(40%~80%)，科技型前期极高(60%~120%)后期下降。
5. **息税前利润率 (EBIT Margin)**：合理给出预测，如处于亏损可给出未来扭亏为盈路径或如实填负数。
6. **预测期长度（极度重要）**：用户设定的预测年数为 **{proj_years}年**。你必须严格按照此年数生成 `scenario` 中的三个数组（revenue_growth_rates, ebit_margin_rates, reinvestment_rate），每个数组的元素个数必须**刚好等于 {proj_years}**！

--- 原始输入信息 ---
公司名称: {data.get('company_name', '')}
公司描述: {data.get('company_description', '')}
财务报表数据: {data.get('financial_statements', '')}
额外指导: {data.get('additional_guidance', '')}
预测年数: {proj_years}

--- 必须严格返回的 JSON 格式 ---
```json
{{
    "_comment": "简要说明你对该公司行业属性的判断。如有严重缺失、亏损或强周期特征，请务必以【警告：...】开头强烈提示。",
    "current_financials": {{
        "revenue": 150000.0, 
        "tax_rate": 0.15,
        "total_debt": 30000.0,
        "cash_and_equivalents": 12000.0
    }},
    "valuation_assumptions": {{
        "projection_years": {proj_years},
        "discount_rate": 0.10,
        "terminal_growth_rate": 0.02
    }},
    "scenario": {{
        "revenue_growth_rates": {example_growth},
        "ebit_margin_rates": {example_margin},
        "reinvestment_rate": {example_reinv}
    }}
}}
```
"""
    try:
        completion = client.chat.completions.create(
            model="deepseek-v3.2-exp",
            messages=[{'role': 'user', 'content': prompt}],
            extra_body={"enable_thinking": True},
            stream=True,
            max_tokens=2048,
            temperature=0.7,
        )
        res = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                res += chunk.choices[0].delta.content

        if '```json' in res:
            json_str = res.split('```json')[1].split('```')[0].strip()
        else:
            json_str = res.strip()
        return jsonify({"status": "success", "data": json.loads(json_str)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json
    try:
        cf = data['current_financials']
        if float(cf.get('revenue', 0)) <= 0:
            return jsonify({"status": "error",
                            "message": "营业总收入(Revenue)为 0 或未提供，严重缺少核心财报数据，无法进行计算！请检查或手动补充数据。"})

        va = data['valuation_assumptions']
        sc = data['scenario']
        rev_growth = [float(x) for x in str(sc['revenue_growth_rates']).split(',')]
        ebit_margin = [float(x) for x in str(sc['ebit_margin_rates']).split(',')]
        reinv_rate = [float(x) for x in str(sc['reinvestment_rate']).split(',')]

        result = calculate_dcf_valuation(
            float(cf['revenue']), int(va['projection_years']), rev_growth,
            ebit_margin, float(cf['tax_rate']), reinv_rate, float(va['discount_rate']),
            float(va['terminal_growth_rate'])
        )
        ev = result['enterprise_value']
        eq = ev - float(cf['total_debt']) + float(cf['cash_and_equivalents'])

        return jsonify({"status": "success", "enterprise_value": ev, "equity_value": eq,
                        "details": result['details'].to_dict('records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/calculate_reverse', methods=['POST'])
def calculate_reverse():
    data = request.json
    try:
        cf = data['current_financials']
        market_cap = float(data.get('current_market_cap', 0))
        if float(cf.get('revenue', 0)) <= 0: return jsonify(
            {"status": "error", "message": "营业总收入(Revenue)为 0 或未提供，无法计算隐含预期。"})
        if market_cap <= 0: return jsonify({"status": "error", "message": "目标市值不能小于或等于0！"})

        va = data['valuation_assumptions']
        sc = data['scenario']
        result = calculate_reverse_dcf(
            market_cap, float(cf['revenue']), int(va['projection_years']),
            [float(x) for x in str(sc['ebit_margin_rates']).split(',')], float(cf['tax_rate']),
            [float(x) for x in str(sc['reinvestment_rate']).split(',')], float(va['discount_rate']),
            float(va['terminal_growth_rate']), float(cf['total_debt']), float(cf['cash_and_equivalents'])
        )
        return jsonify({"status": "success", "implied_g": result['implied_g'], "msg": result['msg'],
                        "details": result['details'].to_dict('records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==============================================================================
#  前端 HTML 模板
# ==============================================================================
BASE_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>AI 智能估值系统 (DCF & Reverse DCF)</title>
    <!-- 引入 Bootstrap 与 Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    <style>
        body { background-color: #f8f9fa; font-size: 0.95rem; }
        .card { margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .card-header { font-weight: bold; background-color: #e9ecef; }
        .form-label { font-weight: 500; font-size: 0.85rem; color: #555; margin-bottom: 0.2rem; }
        .result-box { background-color: #e3f2fd; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
        .result-number { font-size: 1.5rem; font-weight: bold; color: #0d6efd; }
        textarea.code-font { font-family: monospace; font-size: 0.85rem; }
    </style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4 shadow-sm">
  <div class="container-fluid">
    <a class="navbar-brand fw-bold text-warning" href="/">📈 智能投研系统</a>
    <div class="collapse navbar-collapse">
      <ul class="navbar-nav me-auto">
        <li class="nav-item"><a class="nav-link {% if active_page == 'dcf' %}active{% endif %}" href="/">▶️ 正向 DCF 估值</a></li>
        <li class="nav-item"><a class="nav-link {% if active_page == 'reverse' %}active{% endif %}" href="/reverse">◀️ 反向 DCF (隐含增长率)</a></li>
      </ul>
    </div>
  </div>
</nav>

<div class="container-fluid px-4">
    {% block content %}{% endblock %}
</div>

<!-- Toast -->
<div class="position-fixed bottom-0 end-0 p-3" style="z-index: 11">
  <div id="liveToast" class="toast text-white bg-success border-0" role="alert">
    <div class="d-flex">
      <div class="toast-body" id="toast-msg">操作成功</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
    window.dcf_warning_msg = null;
    function showToast(msg, isError=false) {
        const toastEl = document.getElementById('liveToast');
        document.getElementById('toast-msg').innerText = msg;
        toastEl.className = `toast text-white border-0 ${isError ? 'bg-danger' : 'bg-success'}`;
        new bootstrap.Toast(toastEl).show();
    }
    function formatMoney(val) {
        if (Math.abs(val) >= 10000) return (val / 10000).toLocaleString('zh-CN', {minimumFractionDigits:2, maximumFractionDigits:2}) + ' <span class="fs-6">亿元</span>';
        return val.toLocaleString('zh-CN', {minimumFractionDigits:2, maximumFractionDigits:2}) + ' <span class="fs-6">万元</span>';
    }
</script>
{% block scripts %}{% endblock %}
</body>
</html>
"""

DCF_CONTENT = r"""
{% extends "base.html" %}
{% block content %}
<h4 class="mb-3">正向 DCF 模型 (推导目标市值)</h4>
<div class="row">
    <!-- 左侧 AI 输入与字典 -->
    <div class="col-md-4">
        <div class="card border-primary">
            <div class="card-header bg-primary text-white">1. 数据自动获取与 AI 提取</div>
            <div class="card-body">

                <div class="input-group mb-3 shadow-sm">
                    <input type="text" class="form-control form-control-sm" id="ai-name" placeholder="公司名称或6位代码 (如: 000001)">
                    <button class="btn btn-sm btn-success fw-bold" id="btn-auto-fetch"><i class="bi bi-cloud-arrow-down"></i> 自动提取财报</button>

                    <span class="input-group-text bg-light text-muted" style="font-size: 0.85rem;"><i class="bi bi-calendar-range me-1"></i>设定预测年数</span>
                    <input type="number" class="form-control form-control-sm text-center fw-bold text-primary" id="ai-years" value="5" min="1" max="20" style="flex: 0 0 70px;">
                </div>

                <div class="mb-3">
                    <label for="ai-desc" class="form-label text-muted small mb-1">公司业务与描述</label>
                    <textarea class="form-control form-control-sm shadow-sm" id="ai-desc" rows="2" placeholder="自动提取后将在此填充..."></textarea>
                </div>

                <div class="mb-3">
                    <label for="ai-financials" class="form-label text-muted small mb-1">核心财报数据</label>
                    <textarea class="form-control form-control-sm shadow-sm" id="ai-financials" rows="3" placeholder="自动提取后将在此填充..."></textarea>
                </div>

                <div class="mb-3">
                    <label for="ai-guidance" class="form-label text-muted small mb-1">额外指导意见 (选填)</label>
                    <textarea class="form-control form-control-sm shadow-sm" id="ai-guidance" rows="2" placeholder="如：假设未来三年营收不增长..."></textarea>
                </div>

                <button class="btn btn-primary w-100 fw-bold shadow-sm" id="btn-ai-generate"><i class="bi bi-robot"></i> 交给 AI 分析并构建参数字典</button>
            </div>
        </div>

        <div class="card">
            <div class="card-header">参数字典互通区</div>
            <div class="card-body">
                <textarea class="form-control code-font mb-2" id="dict-textarea" rows="7" placeholder="JSON字典..."></textarea>
                <div class="d-flex justify-content-between">
                    <button class="btn btn-sm btn-outline-success" onclick="importDict()">⬆️ 导入面板</button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="exportDict()">⬇️ 生成字典</button>
                    <button class="btn btn-sm btn-outline-dark" onclick="copyDict()">📋 复制</button>
                </div>
            </div>
        </div>
    </div>

    <!-- 中间 参数编辑 -->
    <div class="col-md-4">
        <div class="card border-info">
            <div class="card-header">2. 估值核心假设与参数微调 <span class="badge bg-danger ms-2">单位统一: 万元</span></div>
            <div class="card-body">
                <div class="row mb-2">
                    <div class="col-6"><label class="form-label">基期营收 (万元)</label><input type="number" class="form-control form-control-sm" id="param-revenue" value="100000"></div>
                    <div class="col-6"><label class="form-label">有效税率</label><input type="number" step="0.01" class="form-control form-control-sm" id="param-tax" value="0.15"></div>
                    <div class="col-6"><label class="form-label">总有息负债 (万元)</label><input type="number" class="form-control form-control-sm" id="param-debt" value="20000"></div>
                    <div class="col-6"><label class="form-label">货币资金 (万元)</label><input type="number" class="form-control form-control-sm" id="param-cash" value="35000"></div>
                </div>
                <div class="row mb-2 mt-3">
                    <div class="col-4"><label class="form-label">预测年数</label><input type="number" class="form-control form-control-sm bg-light" id="param-years" value="5" readonly></div>
                    <div class="col-4"><label class="form-label">折现率 WACC</label><input type="number" step="0.01" class="form-control form-control-sm" id="param-discount" value="0.10"></div>
                    <div class="col-4"><label class="form-label">永续增长率</label><input type="number" step="0.01" class="form-control form-control-sm" id="param-term-growth" value="0.02"></div>
                </div>
                <div class="mb-2 mt-3"><label class="form-label">营收增长率 (用逗号分隔)</label><input type="text" class="form-control form-control-sm code-font" id="param-rev-growth" value="0.15, 0.12, 0.10, 0.08, 0.05"></div>
                <div class="mb-2"><label class="form-label">EBIT 利润率 (用逗号分隔)</label><input type="text" class="form-control form-control-sm code-font" id="param-ebit-margin" value="0.10, 0.12, 0.13, 0.15, 0.15"></div>
                <div class="mb-3"><label class="form-label">再投资率 (用逗号分隔，注意轻/重资产区别)</label><input type="text" class="form-control form-control-sm code-font" id="param-reinv-rate" value="0.3, 0.25, 0.2, 0.15, 0.15"></div>

                <button class="btn btn-info text-white w-100 btn-lg shadow-sm" onclick="calcVal()">⚡ 开始折现计算</button>
            </div>
        </div>
    </div>

    <!-- 右侧 结果 -->
    <div class="col-md-4">
        <div class="card border-success">
            <div class="card-header bg-success text-white">3. 估值结果</div>
            <div class="card-body">
                <div class="result-box"><div class="text-muted small">企业价值 (Enterprise Value)</div><div class="result-number" id="res-ev">0.00</div></div>
                <div class="result-box bg-success text-white bg-opacity-10"><div class="text-muted small">合理股权市值 (Equity Value)</div><div class="result-number text-success" id="res-eq">0.00</div></div>
                <div class="text-muted small"><i class="bi bi-info-circle"></i> 股权市值 = 企业价值 - 总负债 + 现金及现金等价物</div>
            </div>
        </div>
    </div>
</div>

<div class="card mt-2"><div class="card-header">现金流预测明细 (单位: 万元)</div><div class="card-body p-0"><table class="table table-striped text-center mb-0" id="res-table"><thead><tr class="table-dark"><th>年份</th><th>营收</th><th>EBIT</th><th>NOPAT</th><th>再投资额</th><th>自由现金流 FCFF</th><th>折现 FCFF</th></tr></thead><tbody></tbody></table></div></div>
{% endblock %}

{% block scripts %}
<script>
    document.getElementById('btn-auto-fetch').onclick = async function() {
        const query = document.getElementById('ai-name').value.trim();
        if (!query) {
            showToast("请输入公司名称或6位股票代码！", true);
            return;
        }

        this.disabled = true; 
        this.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 正在云端获取...`;
        showToast("正在通过 Tushare 获取最新年报及基本面数据，请稍等...");

        try {
            const res = await fetch('/api/autofetch', { 
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query: query })
            });
            const d = await res.json();
            if (d.status === 'success') { 
                document.getElementById('ai-name').value = d.company_name;
                document.getElementById('ai-desc').value = d.company_desc;
                document.getElementById('ai-financials').value = d.financials;
                showToast("✅ 财报及基本信息提取成功！可以修改后交给AI。"); 
            } else {
                showToast(d.message, true);
            }
        } catch(e) { 
            showToast("请求错误，请检查网络或后端日志", true); 
        }
        this.disabled = false; 
        this.innerHTML = '<i class="bi bi-cloud-arrow-down"></i> 自动提取财报';
    };

    function buildDict() {
        return {
            current_financials: { revenue: Number(document.getElementById('param-revenue').value), tax_rate: Number(document.getElementById('param-tax').value), total_debt: Number(document.getElementById('param-debt').value), cash_and_equivalents: Number(document.getElementById('param-cash').value) },
            valuation_assumptions: { projection_years: Number(document.getElementById('param-years').value), discount_rate: Number(document.getElementById('param-discount').value), terminal_growth_rate: Number(document.getElementById('param-term-growth').value) },
            scenario: { revenue_growth_rates: document.getElementById('param-rev-growth').value.replace(/\s+/g, ''), ebit_margin_rates: document.getElementById('param-ebit-margin').value.replace(/\s+/g, ''), reinvestment_rate: document.getElementById('param-reinv-rate').value.replace(/\s+/g, '') }
        };
    }

    function parseDict(dict) {
        document.getElementById('param-revenue').value = dict.current_financials.revenue || 0; 
        document.getElementById('param-tax').value = dict.current_financials.tax_rate || 0;
        document.getElementById('param-debt').value = dict.current_financials.total_debt || 0; 
        document.getElementById('param-cash').value = dict.current_financials.cash_and_equivalents || 0;

        const years = dict.valuation_assumptions.projection_years || 5;
        document.getElementById('param-years').value = years; 
        document.getElementById('ai-years').value = years; // 互相联动同步

        document.getElementById('param-discount').value = dict.valuation_assumptions.discount_rate || 0.1;
        document.getElementById('param-term-growth').value = dict.valuation_assumptions.terminal_growth_rate || 0.02;

        document.getElementById('param-rev-growth').value = dict.scenario.revenue_growth_rates.join(', ');
        document.getElementById('param-ebit-margin').value = dict.scenario.ebit_margin_rates.join(', '); 
        document.getElementById('param-reinv-rate').value = dict.scenario.reinvestment_rate.join(', ');

        window.dcf_warning_msg = null; 
        if(dict._comment) {
            if (dict._comment.includes("警告")) {
                window.dcf_warning_msg = dict._comment;
                alert("⚠️ AI 智能分析预警：\n\n" + dict._comment);
            } else {
                showToast("AI 分析评语: " + dict._comment);
            }
        }
    }

    function exportDict() {
        const d = buildDict();
        d.scenario.revenue_growth_rates = d.scenario.revenue_growth_rates.split(',').map(Number);
        d.scenario.ebit_margin_rates = d.scenario.ebit_margin_rates.split(',').map(Number);
        d.scenario.reinvestment_rate = d.scenario.reinvestment_rate.split(',').map(Number);
        document.getElementById('dict-textarea').value = JSON.stringify(d, null, 4); showToast("字典生成成功");
    }

    function importDict() { try { parseDict(JSON.parse(document.getElementById('dict-textarea').value)); showToast("导入成功"); } catch(e) { showToast("JSON无效", true); } }
    function copyDict() { navigator.clipboard.writeText(document.getElementById('dict-textarea').value); showToast("已复制"); }

    document.getElementById('btn-ai-generate').onclick = async function() {
        this.disabled = true; this.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 正在让AI深度分析中...';
        try {
            const res = await fetch('/api/generate', { method: 'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({ 
                    company_name: document.getElementById('ai-name').value, 
                    company_description: document.getElementById('ai-desc').value, 
                    financial_statements: document.getElementById('ai-financials').value, 
                    additional_guidance: document.getElementById('ai-guidance').value,
                    projection_years: document.getElementById('ai-years').value 
                })
            });
            const d = await res.json();
            if(d.status === 'success') { 
                document.getElementById('dict-textarea').value = JSON.stringify(d.data, null, 4); 
                parseDict(d.data); 
                showToast("AI 解析推演成功！"); 
            } else { showToast(d.message, true); }
        } catch(e) { showToast("请求错误", true); }
        this.disabled = false; this.innerHTML = '<i class="bi bi-robot"></i> 交给 AI 分析并构建参数字典';
    };

    async function calcVal() {
        if (window.dcf_warning_msg) {
            const proceed = confirm("🚨 安全拦截提示：\n\n" + window.dcf_warning_msg + "\n\n强行套用模型可能会得出极其荒谬的估值。\n您确定要【强制进行计算】吗？");
            if (!proceed) { showToast("已取消计算", true); return; }
        }
        const payload = buildDict();
        const res = await fetch('/api/calculate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
        const data = await res.json();
        if(data.status==='success') {
            document.getElementById('res-ev').innerHTML = formatMoney(data.enterprise_value);
            document.getElementById('res-eq').innerHTML = formatMoney(data.equity_value);
            const tb = document.querySelector('#res-table tbody'); tb.innerHTML = '';
            data.details.forEach(r => {
                tb.innerHTML += `<tr><td>${r.Year}</td><td>${formatMoney(r.Revenue)}</td><td>${formatMoney(r.EBIT)}</td><td>${formatMoney(r.NOPAT)}</td><td>${formatMoney(r.Reinvestment)}</td><td class="text-primary fw-bold">${formatMoney(r.FCFF)}</td><td class="text-success">${formatMoney(r['PV of FCFF'])}</td></tr>`;
            });
            showToast("计算完成");
        } else {
            alert("❌ 计算失败：\n" + data.message);
            showToast("计算被阻断", true);
        }
    }
    window.onload = exportDict;
</script>
{% endblock %}
"""

REVERSE_DCF_CONTENT = r"""
{% extends "base.html" %}
{% block content %}
<h4 class="mb-3 text-warning">反向 DCF 模型 (探索市场隐含预期)</h4>
<div class="row">
    <div class="col-md-5">
        <div class="card border-warning">
            <div class="card-header bg-warning text-dark fw-bold">输入已知财务参数 <span class="badge bg-danger ms-2">单位统一: 万元</span></div>
            <div class="card-body">
                <div class="p-3 bg-warning bg-opacity-10 border border-warning rounded mb-3 shadow-sm">
                    <label class="form-label fw-bold text-dark fs-6">🎯 当前实际市值 (Current Market Cap) [万元]</label>
                    <input type="number" class="form-control form-control-lg text-primary fw-bold" id="param-market-cap" value="300000">
                    <small class="text-muted">输入当前股票市场的总市值，算法将逆推需要多高的增速才能支撑该市值。</small>
                </div>

                <div class="row mb-2">
                    <div class="col-6"><label class="form-label">基期营收</label><input type="number" class="form-control form-control-sm" id="param-revenue" value="100000"></div>
                    <div class="col-6"><label class="form-label">税率</label><input type="number" step="0.01" class="form-control form-control-sm" id="param-tax" value="0.15"></div>
                    <div class="col-6"><label class="form-label">总有息负债</label><input type="number" class="form-control form-control-sm" id="param-debt" value="20000"></div>
                    <div class="col-6"><label class="form-label">货币资金</label><input type="number" class="form-control form-control-sm" id="param-cash" value="35000"></div>
                </div>

                <div class="row mb-2 mt-3">
                    <div class="col-4"><label class="form-label">折现率 WACC</label><input type="number" step="0.01" class="form-control form-control-sm" id="param-discount" value="0.10"></div>
                    <div class="col-4"><label class="form-label">预测年数</label><input type="number" class="form-control form-control-sm" id="param-years" value="5"></div>
                    <div class="col-4"><label class="form-label">永续增长率</label><input type="number" step="0.01" class="form-control form-control-sm" id="param-term-growth" value="0.02"></div>
                </div>

                <div class="mb-2 mt-3"><label class="form-label">EBIT 利润率预测 (逗号分隔)</label><input type="text" class="form-control form-control-sm code-font" id="param-ebit-margin" value="0.10, 0.12, 0.13, 0.15, 0.15"></div>
                <div class="mb-3"><label class="form-label">再投资率预测 (逗号分隔)</label><input type="text" class="form-control form-control-sm code-font" id="param-reinv-rate" value="0.3, 0.25, 0.2, 0.15, 0.15"></div>

                <button class="btn btn-warning w-100 btn-lg fw-bold shadow-sm" onclick="calcReverse()"><i class="bi bi-search"></i> 🔍 开始二分查找运算</button>
            </div>
        </div>
    </div>
    <div class="col-md-7">
        <div class="card border-primary">
            <div class="card-header bg-primary text-white">隐含预期结果分析</div>
            <div class="card-body">
                <div class="result-box bg-dark text-white p-4 text-center mb-3 shadow-sm">
                    <div class="text-white-50 fs-5 mb-2">市场价格隐含的年复合营收增长率 (Implied CAGR)</div>
                    <div class="result-number text-warning" style="font-size: 3.5rem;" id="res-implied-g">0.00%</div>
                    <div id="res-msg" class="mt-2 text-info">等待计算...</div>
                </div>

                <div class="table-responsive">
                    <table class="table table-sm table-striped text-center mb-0" id="res-table">
                        <thead class="table-dark"><tr><th>年份</th><th>营收</th><th>EBIT</th><th>NOPAT</th><th>再投资额</th><th>FCFF</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    async function calcReverse() {
        const payload = {
            current_market_cap: Number(document.getElementById('param-market-cap').value),
            current_financials: { revenue: Number(document.getElementById('param-revenue').value), tax_rate: Number(document.getElementById('param-tax').value), total_debt: Number(document.getElementById('param-debt').value), cash_and_equivalents: Number(document.getElementById('param-cash').value) },
            valuation_assumptions: { projection_years: Number(document.getElementById('param-years').value), discount_rate: Number(document.getElementById('param-discount').value), terminal_growth_rate: Number(document.getElementById('param-term-growth').value) },
            scenario: { ebit_margin_rates: document.getElementById('param-ebit-margin').value.replace(/\s+/g, ''), reinvestment_rate: document.getElementById('param-reinv-rate').value.replace(/\s+/g, '') }
        };

        const res = await fetch('/api/calculate_reverse', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
        const data = await res.json();

        if(data.status==='success') {
            document.getElementById('res-implied-g').innerText = (data.implied_g * 100).toFixed(2) + '%';
            document.getElementById('res-msg').innerText = data.msg;
            const tb = document.querySelector('#res-table tbody'); tb.innerHTML = '';
            data.details.forEach(r => {
                tb.innerHTML += `<tr><td>${r.Year}</td><td>${formatMoney(r.Revenue)}</td><td>${formatMoney(r.EBIT)}</td><td>${formatMoney(r.NOPAT)}</td><td>${formatMoney(r.Reinvestment)}</td><td class="text-primary fw-bold">${formatMoney(r.FCFF)}</td></tr>`;
            });
            showToast("反向推演完成！");
        } else {
            alert("❌ 计算失败：\n" + data.message);
            showToast("计算被阻断", true);
        }
    }
</script>
{% endblock %}
"""


def render_page(template_content, active_page):
    full_template = "{% extends 'base.html' %}" + template_content.replace('{% extends "base.html" %}', '')
    app.jinja_env.loader = app.jinja_env.loader or {}
    import jinja2
    app.jinja_loader = jinja2.DictLoader({'base.html': BASE_HTML})
    return render_template_string(full_template, active_page=active_page)


@app.route('/')
def index(): return render_page(DCF_CONTENT, active_page='dcf')


@app.route('/reverse')
def reverse_dcf(): return render_page(REVERSE_DCF_CONTENT, active_page='reverse')


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 双引擎智能投研系统启动完成！")
    print("👉 请在浏览器打开: http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
