library(ggplot2)
data(Orange)

p <- ggplot(Orange, aes(x=age, y=circumference, color=Tree)) + geom_line(size = 0.8) + geom_point() + theme_classic() + theme(legend.position = "none")
p <- p + xlab("Days") + ylab("Circumference (mm)")
print(p)

ggsave("C:/Users/ville/Desktop/master/reports/code/ME.png",p)
