public class Totient {

    final static int size = 360000000;
    final static int[] phi = new int[size];

    public static void main(String[] args) {
        long time = System.currentTimeMillis();
        long sum = 0;

        phi[1] = 1;
        for (int i = 2; i < size; i++) {
            if (phi[i] == 0) {
                phi[i] = i - 1;
                for (int j = 2; i * j < size; j++) {
                    if (phi[j] == 0)
                        continue;

                    int q = j;
                    int f = i - 1;
                    while (q % i == 0) {
                        f *= i;
                        q /= i;
                    }
                    phi[i * j] = f * phi[q];
                }
            }
            sum += phi[i];
        }
        System.out.println(System.currentTimeMillis() - time);
        System.out.println(sum);
    }
}
