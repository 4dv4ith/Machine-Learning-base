from sympy import symbols,Eq,solve

x,y = symbols('x y')

eq1 = Eq(x+y,2200)

eq2 = Eq(1.5*x + 4.0*y,5050)

solution = solve((eq1,eq2),(x,y))

Number_of_children = solution[x]
Number_of_adult = solution[y]

print(f"Number of children = {Number_of_children}")
print(f"Number of Adult = {Number_of_adult}")