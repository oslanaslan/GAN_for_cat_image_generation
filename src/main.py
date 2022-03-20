from CatGAN import CatGan

if __name__ == '__main__':

    gan = CatGan()
    gan.loadData()
    # gan.loadModel('GAN_torch_ngf=80_epoch=100_lr=5e-05_beta=0.2.pth', ngf=80)
    # print('Data loaded')
    #
    # gan.getGenImg(show=False, count=10)
    #
    # print('Img')
    #
    # for i in range(2):
    #     img_list, G_losses, D_losses = gan.train(5)
    #     gan.plotRes(img_list)
    #     print(gan.fidScore(pic_num=100))
    #
