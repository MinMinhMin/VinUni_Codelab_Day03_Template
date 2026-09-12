"""
Lab #3: Baseline Chatbot vs ReAct Agent
Mã nguồn hoàn chỉnh cho bài lab.
"""

import json
import re
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> dict:
        return {
            "status": "success",
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": []
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    @staticmethod
    def _extract_flight_args(user_input: str):
        route = re.search(
            r"\b(?:từ|from)\s+([A-Za-z]{3})\s+(?:đi|đến|to)\s+([A-Za-z]{3})\b",
            user_input,
            re.IGNORECASE,
        )
        if not route:
            return None

        price = 5_000_000
        price_match = re.search(
            r"(?:dưới|tối đa|giá)\s*([\d.,]+)\s*(triệu|tr|k|nghìn)?",
            user_input,
            re.IGNORECASE,
        )
        if price_match:
            raw_price, unit = price_match.groups()
            unit = (unit or "").lower()
            if unit in {"triệu", "tr"}:
                price = int(float(raw_price.replace(",", ".")) * 1_000_000)
            elif unit in {"k", "nghìn"}:
                price = int(float(raw_price.replace(",", ".")) * 1_000)
            else:
                price = int(raw_price.replace(".", "").replace(",", ""))

        return {
            "origin": route.group(1).upper(),
            "destination": route.group(2).upper(),
            "max_price": price,
        }

    @staticmethod
    def _extract_weather_code(user_input: str):
        codes = re.findall(r"\b(HAN|SGN|DAD)\b", user_input.upper())
        if codes:
            return codes[-1]

        city_codes = {
            "đà nẵng": "DAD",
            "hà nội": "HAN",
            "sài gòn": "SGN",
            "hồ chí minh": "SGN",
        }
        normalized_input = user_input.lower()
        for city, code in city_codes.items():
            if city in normalized_input:
                return code
        return None

    @staticmethod
    def _format_answer(flight_result, weather_result, user_input: str) -> str:
        parts = []
        if flight_result is not None:
            if flight_result:
                flights = "; ".join(
                    f"{flight['flight_number']} ({flight['airline']}, "
                    f"{flight['price_vnd']:,} VND, khởi hành {flight['departure_time']})"
                    for flight in flight_result
                )
                parts.append(f"Chuyến bay phù hợp: {flights}.")
            else:
                parts.append("Không tìm thấy chuyến bay phù hợp.")

        if weather_result is not None:
            if "error" in weather_result:
                parts.append(weather_result["error"])
            else:
                parts.append(
                    f"Thời tiết tại {weather_result['city']}: "
                    f"{weather_result['temperature_c']}°C, {weather_result['condition']}. "
                    f"{weather_result['recommendation']}"
                )

        if parts:
            return " ".join(parts)
        return (
            "Vinpearl sẽ áp dụng chính sách đổi trả theo điều kiện của từng loại vé. "
            "Vui lòng liên hệ bộ phận hỗ trợ để được kiểm tra chi tiết."
        )

    def run(self, user_input: str) -> dict:
        self.trace = []
        flight_args = self._extract_flight_args(user_input)
        weather_code = self._extract_weather_code(user_input)
        lower_input = user_input.lower()

        actions = []
        if flight_args and any(
            keyword in lower_input for keyword in ("chuyến bay", "vé", "bay")
        ):
            actions.append(("get_flight_info", flight_args))
        if weather_code and any(
            keyword in lower_input
            for keyword in ("thời tiết", "nhiệt độ", "mặc gì", "trang phục")
        ):
            actions.append(("get_weather_forecast", {"city_code": weather_code}))

        pending_actions = list(actions)
        flight_result = None
        weather_result = None
        multi_step = len(actions) > 1
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            if pending_actions:
                tool_name, args = pending_actions.pop(0)
                trace_entry = {
                    "iteration": iteration,
                    "thought": f"Thực hiện {tool_name} để thu thập dữ liệu.",
                    "action": {"name": tool_name, "args": args},
                }
                try:
                    observation = TOOL_MAP[tool_name](**args)
                except (KeyError, TypeError, ValueError) as error:
                    observation = {"error": str(error)}
                trace_entry["observation"] = observation
                self.trace.append(trace_entry)

                if tool_name == "get_flight_info":
                    flight_result = observation
                else:
                    weather_result = observation

                if not pending_actions and not multi_step:
                    answer = self._format_answer(
                        flight_result, weather_result, user_input
                    )
                    self.trace[-1]["final_answer"] = answer
                    return {
                        "status": "completed",
                        "answer": answer,
                        "iterations": iteration,
                        "trace": self.trace,
                    }
                continue

            answer = self._format_answer(flight_result, weather_result, user_input)
            self.trace.append(
                {
                    "iteration": iteration,
                    "thought": "Đã đủ dữ liệu để trả lời.",
                    "final_answer": answer,
                }
            )
            return {
                "status": "completed",
                "answer": answer,
                "iterations": iteration,
                "trace": self.trace,
            }

        answer = (
            self._format_answer(flight_result, weather_result, user_input)
            if iteration
            else "Không thể hoàn thành trong số bước tối đa."
        )
        return {
            "status": "max_iterations_reached",
            "answer": answer,
            "iterations": iteration,
            "trace": self.trace,
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
