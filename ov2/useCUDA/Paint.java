
import java.awt.*;
import javax.swing.*;

public class Paint extends JPanel {
    int r =20; // object size
    int startAngle; // current velocity angle
    int arcAngle = 30 ; //  arc angle 
    double cvAngle; // current velocity Angle
    int x,y;
    
    public void paintComponent(Graphics g){
	super.paintComponent(g);

	/* set background color */
	g.setColor(Color.white);
	Dimension dim = getSize();
	g.fillRect(0 , 0 , dim.width , dim.height); 

	/* set simulation time indicator */
	g.setColor(Color.black);
	g.drawString("Simulation Time = "+Float.toString((float)Solve.time) , 10, 10);		

	//main

	/* the object of number 0 is colored black */
	x=(int)((double)dim.width * Solve.x[0]/Solve.Lx);
	y=dim.height - (int)((double)dim.height*Solve.y[0]/Solve.Ly); //左下の座標を（0,0）とする
	cvAngle = Math.atan2(Solve.vy[0],Solve.vx[0]);
	startAngle = (int)(Math.toDegrees(cvAngle))-arcAngle/2+180;
	g.fillArc(x-r, y+r, 2*r , 2*r, startAngle, arcAngle);


	g.setColor(Color.blue);
	for (int i=1;i<Solve.N;i++){
	    x=(int)((double)dim.width * Solve.x[i]/Solve.Lx);
	    y=dim.height - (int)((double)dim.height*Solve.y[i]/Solve.Ly); //左下の座標を（0,0）とする
	    cvAngle = Math.atan2(Solve.vy[i],Solve.vx[i]);
	    startAngle = (int)(Math.toDegrees(cvAngle)) - arcAngle/2+180 ;
	    g.fillArc(x-r, y+r , 2*r , 2*r,startAngle,arcAngle);
	}
    }
}	