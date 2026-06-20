from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"




# ------------------ Tools (langchian @tool decorator) ------------------



@tool
def get_product_price(product: str) -> float:
    """ Lookup the price of a product in the catalog. """
    print(f"       >>  Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    # what is this below function doing? It is looking up the price of a product in the catalog. If the product is not found, it returns 0.0.
    return prices.get(product.lower(), 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """ Apply a discount to a price based on the discount tier. """
    print(f"       >>  Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discounts = {"gold": 0.20, "silver": 0.10, "bronze": 0.05}
    discount = discounts.get(discount_tier.lower(), 0.0)
    return price * (1 - discount)


# ------------------- Agent Loop -------------------

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    # Below Example of above tools_dict 
    # tools_dict = {
    #     "hammer": hammer,
    #     "screwdriver": screwdriver
    # }
    # Now if someone says: "Use screwdriver" so you can quickly do: tools_dict["screwdriver"] instead of looping through the whole list, so "Make it easy to find a tool by its name when the LLM asks for it."
    llm = init_chat_model(f"openai:gpt-4o-mini", temperature=0)
    # let model know which tools we have here is the below
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    # ReAct Loop (Agent Loop or Agentic loop)
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n------------Iteration {iteration} -------- ")

        ai_message = llm_with_tools.invoke(messages)
        # breakpoint()

        tool_calls = ai_message.tool_calls

        # if no tool calls, this is the final answer
        if not tool_calls:
            print(f"\n Final answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get('name')
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get('id')

        print(f".     [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"tool {tool_to_use} not found")
        
        observation = tool_to_use.invoke(tool_args)

        # Till here our llm just collected the tool just check image algorithm till here ./image.png.  -> For this process till here cal it as Reasoning(ReAct agent)
        print(f"     [Tool result].   {observation}")


        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print("ERROR: Max iterations reached without a final answer")
    return None






if __name__ == "__main__":
    print("Hello langchain agent (.blind_tools)")
    result = run_agent("what is the price of the laptop after applying a gold discount")



