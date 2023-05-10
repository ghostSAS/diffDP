x0 = [1,1,1,1];
u = 1;
disp(cart_pen(0,x0,u))

ode45(@(t,y)cart_pen(t,x,1), .05)

function F = cart_pen(t,x,u)
    xi1 = x(1);
    xi2 = x(2);
    xi3 = x(3);
    xi4 = x(4);
    xi5 = u;
    
    m1 = .5;
    m2 = 0.2;
    l = 0.3;
    g = 9.81;

    F= [xi3;
        xi4;
        1/(m1+m2*(1-cos(xi2)^2))*(l*m2*sin(xi2)*xi4^2+xi5+m2*g*cos(xi2)*sin(xi2));
        -1/(l*m1+l*m2*(1-cos(xi2)^2))*(l*m2*cos(xi2)*sin(xi2)*(xi4)^2+xi5*cos(xi2)+(m1+m2)*g*sin(xi2))];
end


    