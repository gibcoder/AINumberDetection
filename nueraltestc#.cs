double weight_1_1, weight_2_1
double weight_1_2, weight_2_2

public interface Classification(double input_1,double input_2){
    double output_1 = (input_1*weight_1_1)+(input_1*weight_2_1)
    double output_1 = (input_1*weight_1_1)+(input_1*weight_2_1)

    if (output_1>output_2)
    {
        return 0
    }
    else
    {
        return 1
    }

}
public void Visualise(double graphX, double graphY){
    int predictedClass=Classify(graphX,graphY);

    if (predictedClass==0){
        graph.SetColour(graphX,graphY,safeColour)

    }
    else if (predictedClass==1){
        graphX.SetColour(graphX,graphY,posionousColour);
    }
}
    public void SetColour(double x, double y, string colour)
    {
        Console.WriteLine($"Setting colour at ({x}, {y}) to {colour}");
    }


class Program
{
    static void Main(string[] args)
    {
        Classifier classifier = new Classifier(1.0, -1.0, 1.0, 1.0);

        // Example visualization
        classifier.Visualise(3.0, 4.0);
        classifier.Visualise(-1.0, 2.0);
    }
}