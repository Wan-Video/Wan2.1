import argparse

def add(x, y):
  """Adds two numbers together."""
  return x + y

def subtract(x, y):
  """Subtracts two numbers."""
  return x - y

def multiply(x, y):
  """Multiplies two numbers."""
  return x * y

def divide(x, y):
  """Divides two numbers."""
  if y == 0:
    raise ValueError("Cannot divide by zero.")
  return x / y

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A simple command-line calculator.")
  parser.add_argument("x", type=float, help="The first number.")
  parser.add_argument("operation", choices=["+", "-", "*", "/"], help="The operation to perform.")
  parser.add_argument("y", type=float, help="The second number.")
  args = parser.parse_args()

  if args.operation == "+":
    result = add(args.x, args.y)
  elif args.operation == "-":
    result = subtract(args.x, args.y)
  elif args.operation == "*":
    result = multiply(args.x, args.y)
  elif args.operation == "/":
    result = divide(args.x, args.y)

  print(f"Result: {result}")
