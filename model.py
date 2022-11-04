from torch import nn
import kornia.geometry.transform


class Block(nn.Module):

    def __init__(self, dim, kernel_size, groups, norm, expansion):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, padding, groups=groups)
        self.norm = nn.BatchNorm2d(dim) if norm else nn.Identity()
        hidden_dim = int(expansion * dim)
        self.ln1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = nn.GELU()
        self.ln2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ln1(out)
        out = self.act(out)
        out = self.ln2(out)
        out += x
        return out

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        def blocks(dim, n, norm=True):
            return [Block(dim, kernel_size=5, groups=1, norm=norm, expansion=2) for _ in range(n)]

        norm = nn.BatchNorm2d

        self.encoder = nn.ModuleList([
                    # 8 x 181 x 560
                    nn.Sequential(
                        nn.Conv2d(8, 16, (3, 3), (1, 2), (1, 1)),
                        norm(16),
                        *blocks(16, n=1),
                        norm(16),
                    ),
                    # 16 x 181 x 280
                    nn.Sequential(
                        nn.Conv2d(16, 32, (3, 3), (1, 2), (1, 1)),
                        norm(32),
                        *blocks(32, n=1),
                        norm(32),
                    ),
                    # 32 x 181 x 140
                    nn.Sequential(
                        nn.Conv2d(32, 64, (3, 3), (1, 2), (1, 1)),
                        norm(64),
                        *blocks(64, n=1),
                        norm(64),
                    ),
                    # 64 x 181 x 70
                    nn.Sequential(
                        nn.Conv2d(64, 128, (3, 3), (1, 2), (1, 1)),
                        norm(128),
                        *blocks(128, n=1),
                        norm(128),
                    ),
                    # 128 x 181 x 35
                    nn.Sequential(
                        nn.Conv2d(128, 128, (4, 3), (4, 1), (1, 1)),
                        norm(128),
                        *blocks(128, n=1),
                        norm(128),
                        # 128 x 45 x 35
                        nn.Conv2d(128, 256, (3, 3), (3, 2), (1, 0)),
                        *blocks(256, n=3),
                        norm(256),
                        # 256 x 15 x 17
                        nn.Conv2d(256, 512, (3, 3), (2, 2), (1, 0)),
                        *blocks(512, n=6),
                        norm(512),
                        # 512 x 8 x 8
                    )
        ])

        self.from_grey = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(2, 8, 1),
                        nn.GELU()
                    ),
                    nn.Sequential(
                        nn.Conv2d(2, 16, 1),
                        nn.GELU()
                    ),
                    nn.Sequential(
                        nn.Conv2d(2, 32, 1),
                        nn.GELU()
                    ),
                    nn.Sequential(
                        nn.Conv2d(2, 64, 1),
                        nn.GELU()
                    ),
                    nn.Sequential(
                        nn.Conv2d(2, 128, 1),
                        nn.GELU()
                    ),])

        self.decoder = nn.ModuleList([
                    # 512 x 8 x 8
                    nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 2, 2, 0),
                        *blocks(256, n=3),
                        #norm(256),
                    # 256 x 16 x 16
                        nn.ConvTranspose2d(256, 128, 2, 2, 0),
                        *blocks(128, n=1),
                        #norm(128),
                    ),
                    # 128 x 32 x 32
                    nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 2, 2, 0),
                        *blocks(64, n=1),
                        #norm(64),
                    ),
                    # 64 x 64 x 64
                    nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 2, 2, 0),
                        *blocks(32, n=1),
                        #norm(32),
                    ),
                    # 32 x 128 x 128
                    nn.Sequential(
                        nn.ConvTranspose2d(32, 16, 2, 2, 0),
                        *blocks(16, n=1),
                        #norm(16),
                    ),
                    # 16 x 256 x 256
                    nn.Sequential(
                        nn.ConvTranspose2d(16, 8, 2, 2, 0),
                        *blocks(8, n=1),
                        #norm(8),
                    ),
                    # 8 x 512 x 512
        ])


        self.to_grey = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(8, 1, 1),
                nn.Sigmoid()
            ),
        ])

    def forward(self, x, angles, step=0, alpha=1):

        """
        step=0: 32x32
        step=1: 64x64
        step=2: 128x128
        step=3: 256x256
        step=4: 512x512
        """

        # encoder
        for i in range(step, -1, -1):
            idx = 5 - (i+1)

            if i == step:
                out = self.from_grey[idx](x)

            out = self.encoder[idx](out)

            if i > 0 and i == step and 0 <= alpha < 1:
                skip_grey = F.avg_pool2d(x, (1,2))
                skip_grey = self.from_grey[idx+1](skip_grey)
                out = (1 - alpha) * skip_grey + alpha * out

        # decoder
        for i in range(step+1):
            x = out
            out = self.decoder[i](x)

            if i == step:
                out_grey = self.to_grey[i](out)

                if i > 0 and 0 <= alpha < 1:
                    skip_grey = F.interpolate(x, scale_factor=2)
                    skip_grey = self.to_grey[i-1](skip_grey)
                    out_grey = (1 - alpha) * skip_grey + alpha * out_grey

        out = out_grey

        out = kornia.geometry.transform.rotate(out, angles)
        return out
